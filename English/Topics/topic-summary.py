#!/usr/bin/env python3
"""topic-summary.py — 2012portal 主題導向兩階段摘要管線（英文版）。

從 topics.txt 讀取主題，每行一個主題：

    Topic[,keyword2,keyword3,...]

第一欄是主題名（也是輸出資料夾名），逗號後是「同時要搜索的內容」。
所有欄位（含主題名本身）都會以 case-insensitive 子串搜尋每篇文章的
標題 + 內文（posts / interview / meeting 三個類別，去重後 ~1795 篇，
與 full-corpus-summary.py 相同的讀取方式）。不含任何關鍵字的文章自動略過。

輸出結構（每個主題）：

    topics/<Topic>/L1/{category}/{folder}.md   L1: 該篇與關鍵字有關的摘要
                                                （只含關鍵字相關內容，其他不放）
    topics/<Topic>/README.md            L2: 全部 L1 摘要合併出的主題總摘要

中斷續跑（resume）：
  - 已存在的 L1 .md 自動略過；已存在的 README.md 自動略過。
  - 依序跑完 topics.txt 所有主題，中斷後重跑同一指令即可從未完成處續。

--layer / --force：
  - 預設 --layer all（不含 --force）：L1、L2 都已有的 .md 全略過。
  - --layer l1 --force：清掉該主題 L1 並全部重做（L2 不動）。
  - --layer l2 --force：依照 prompts-topic/ 的（新）prompt 重跑 L2。
  - --layer all --force：L1、L2 全部重做。
  - 修改 prompts-topic/*.txt 後：L1 改動需 --layer l1/all --force 才生效；
    L2 改動需 --layer l2/all --force 才生效（已產出的 .md 預設不重做）。

token 計算以本地 llama.cpp /tokenize 精確驗證；超 context 預算即拆批。
只讀來源：/rtx-storage/crawler/2012portal 絕不寫入。

Examples:
    # 跑 topics.txt 全部主題（已完成的自動略過）
    python3 topic-summary.py

    # 改完 L2 prompt 後重跑 L2
    python3 topic-summary.py --layer l2 --force

    # 全部重做
    python3 topic-summary.py --layer all --force

    # 小樣本測試（只用前 N 篇來源文章）
    python3 topic-summary.py --limit 200
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_DIR = Path('/storage/workspace/2012portal-tts')
SOURCE_ROOT = Path('/rtx-storage/crawler/2012portal')   # 唯讀
SOURCE_CATEGORIES = ('posts', 'interview', 'meeting')
TOPICS_FILE = BASE_DIR / 'topics.txt'
PROMPT_DIR = BASE_DIR / 'prompts-topic'
TOPICS_DIR = BASE_DIR / 'topics'

DEFAULT_API = os.environ.get('LLAMA_API_URL', 'http://127.0.0.1:8000/v1/chat/completions')
DEFAULT_TOKENIZE_URL = os.environ.get('LLAMA_TOKENIZE_URL', 'http://127.0.0.1:8000/tokenize')
DEFAULT_MODEL = os.environ.get('LLAMA_MODEL', 'qwen')
DEFAULT_CONTEXT = 262144          # llama.cpp n_ctx
CHARS_PER_TOKEN = 3.8             # 英文保守換算 (只用於預估，最終以 /tokenize 為準)

LOG_LINES = []


def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    LOG_LINES.append(line)


# ---------------------------------------------------------------- http helpers

def post_json(url, payload, timeout=2400):
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = Request(url, data=body,
                  headers={'Content-Type': 'application/json'},
                  method='POST')
    with urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def tokenize(text, tokenize_url):
    """回傳 token 數；端點不可用或回傳空陣列時 fail closed。

    注意：此 llama.cpp 版本的 /tokenize 讀取 JSON 的 'content' key
    （舊版讀 'prompt'；送錯 key 會静默回 {"tokens": []}）。"""
    try:
        data = post_json(tokenize_url, {'content': text}, timeout=180)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f'無法使用 /tokenize ({tokenize_url})：{exc}') from exc
    tokens = data.get('tokens')
    if not isinstance(tokens, list):
        raise RuntimeError('/tokenize 回應缺少 tokens 陣列')
    if text and not tokens:
        raise RuntimeError('/tokenize 對非空文字回傳空 tokens 陣列'
                           '（key 不符或端點異常）')
    return len(tokens)


def estimate_tokens(text):
    return int(len(text) / CHARS_PER_TOKEN) + 1


def out_estimate(text):
    """預估單篇文章的「關鍵字摘要」輸出 token 數（prompt 規定只摘關鍵字相關內容）。"""
    n = len(text)
    if n < 400:
        return 80          # 短文章：關鍵字摘要很短
    return min(int(n / CHARS_PER_TOKEN * 0.25), 400)


def llm_call(api_url, model, user_content, max_output, temperature,
             retries=5, timeout=2400, label=''):
    """帶重試的 chat 呼叫。"""
    last = None
    for attempt in range(1, retries + 1):
        try:
            payload = {
                'model': model,
                'messages': [{'role': 'user', 'content': user_content}],
                'temperature': temperature,
                'max_tokens': max_output,
                # Qwen3 thinking 模式會把 max_tokens 吃光在 reasoning_content，
                # 導致 content 為空；摘要任務不需要思考鏈，直接關掉。
                'chat_template_kwargs': {'enable_thinking': False},
            }
            data = post_json(api_url, payload, timeout=timeout)
            content = data['choices'][0]['message'].get('content', '').strip()
            if not content:
                raise RuntimeError('模型回傳空內容')
            return content
        except (HTTPError, URLError, TimeoutError, OSError,
                KeyError, IndexError, json.JSONDecodeError) as exc:
            last = exc
            wait = min(2 ** attempt * 5, 120)
            log(f'  {label} 第 {attempt}/{retries} 次失敗: {exc} → {wait}s 後重試')
            time.sleep(wait)
    raise RuntimeError(f'{label} 推論失敗 (已重試 {retries} 次): {last}')


# ---------------------------------------------------------------- corpus scan

def discover_docs():
    """掃描唯讀來源的三個內容目錄 (posts / interview / meeting)，回傳依 date 排序的
    文章清單。重複內容 (md5 相同) 只保留第一筆；空白文章跳過。
    doc_id 用 `<category>/<folder>` 命名空間化，避免跨類別撞名（有 6 個資料夾名
    同時出現在兩個類別）。hay 為標題+內文的 lowercase，供關鍵字搜尋。"""
    docs, seen_md5 = [], set()
    if not SOURCE_ROOT.is_dir():
        raise RuntimeError(f'來源目錄不存在: {SOURCE_ROOT}')
    for category in SOURCE_CATEGORIES:
        cat_dir = SOURCE_ROOT / category
        if not cat_dir.is_dir():
            log(f'  (類別目錄不存在，略過: {cat_dir})')
            continue
        for folder in sorted(cat_dir.iterdir()):
            if not folder.is_dir():
                continue
            ct = folder / 'content.txt'
            if not ct.exists():
                continue
            text = ct.read_text(encoding='utf-8', errors='replace').strip()
            if not text:
                continue
            md5 = hashlib.md5(text.encode('utf-8', 'replace')).hexdigest()
            if md5 in seen_md5:
                continue
            seen_md5.add(md5)
            date = folder.name.split('__', 1)[0] if '__' in folder.name else '?'
            slug = folder.name.split('__', 1)[1] if '__' in folder.name else folder.name
            title = slug.replace('-', ' ').replace('_', ' ')
            url = ''
            meta = folder / 'meta.json'
            if meta.exists():
                try:
                    m = json.loads(meta.read_text(encoding='utf-8'))
                    date = str(m.get('date') or m.get('date_published') or date)
                    t = m.get('title')
                    if t and t != 'Untitled':
                        title = t
                    url = m.get('url', '')
                except (json.JSONDecodeError, OSError):
                    pass
            doc_id = f'{category}/{folder.name}'
            docs.append({
                'id': doc_id,
                'category': category,
                'folder': folder.name,
                'date': date,
                'year': date[:4] if date[:4].isdigit() else '?',
                'title': title,
                'url': url,
                'text': text,
                'hay': (title + '\n' + text).lower(),
            })
    docs.sort(key=lambda d: (d['date'], d['id']))
    return docs


def doc_section(doc):
    url = f'\nURL：{doc["url"]}' if doc.get('url') else ''
    return (f'\n\n===== FILE {doc["id"]} =====\n'
            f'Title: {doc["title"]}\nDate: {doc["date"]}{url}\n'
            f'--- original ---\n{doc["text"]}')


def parse_topics(path):
    """解析 topics.txt：每行 `Topic[,kw2,kw3...]`。
    回傳 [(topic, [lowercased terms...])]；topic 名稱本身也是搜尋關鍵字。"""
    topics = []
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = [p.strip() for p in re.split(r'[,，]', line) if p.strip()]
        topic = parts[0]
        terms = list(dict.fromkeys([parts[0].lower()] + [p.lower() for p in parts[1:]]))
        topics.append((topic, terms))
    return topics


# ---------------------------------------------------------------- L1: map

def l1_path(topic, doc):
    return TOPICS_DIR / topic / 'L1' / doc['category'] / (doc['folder'] + '.md')


def build_l1_batches(docs, prompt_text, input_budget, out_budget,
                     tokenize_url, max_files):
    """以字數預估貪心打包 (input 與 output 雙預算)；組裝後用 /tokenize
    精確驗證 input，超標即對半拆。"""
    prompt_tokens = tokenize(prompt_text, tokenize_url)
    batches, cur, cur_est, cur_out = [], [], 0, 0
    for doc in docs:
        est = estimate_tokens(doc['text']) + 80
        oest = out_estimate(doc['text'])
        if cur and (cur_est + est + prompt_tokens > input_budget
                    or cur_out + oest > out_budget
                    or len(cur) >= max_files):
            batches.append(cur)
            cur, cur_est, cur_out = [], 0, 0
        cur.append(doc)
        cur_est += est
        cur_out += oest
    if cur:
        batches.append(cur)

    verified = []
    for batch in batches:
        body = prompt_text + ''.join(doc_section(d) for d in batch)
        n = tokenize(body, tokenize_url)
        if n > input_budget:
            if len(batch) > 1:
                half = len(batch) // 2
                log(f'  L1 批次過大 ({n} tokens, {len(batch)} 篇) → 拆成兩批')
                verified.extend(build_l1_batches(
                    batch[:half], prompt_text, input_budget, out_budget,
                    tokenize_url, max_files)
                    + build_l1_batches(
                        batch[half:], prompt_text, input_budget, out_budget,
                        tokenize_url, max_files))
                continue
            log(f'  警告: 單篇文章超過 input 預算 ({n} > {input_budget}): {batch[0]["id"]}')
        verified.append((batch, n))
    return verified


CARD_RE = re.compile(r'^##\s*CARD[:：]\s*(\S+)', re.M)


def parse_cards(output, batch):
    """從批次輸出中切出各篇摘要。回傳 {doc_id: text}。
    允許模型省略 'category/' 前綴或改寫 key → 用 suffix/folder 模糊對齊。"""
    by_key = {}
    for d in batch:
        by_key[d['id']] = d['id']
        by_key[d['folder']] = d['id']
    cards = {}
    matches = list(CARD_RE.finditer(output))
    for i, m in enumerate(matches):
        key = m.group(1).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(output)
        doc_id = by_key.get(key)
        if doc_id is None:
            for d in batch:
                if d['folder'] in key:
                    doc_id = d['id']
                    break
        if doc_id is None:
            continue
        cards[doc_id] = output[m.end():end].strip()
    return cards


def render_prompt(template_path, topic, terms):
    """讀取 prompt 並替換 {topic} / {keywords}（用 replace 而非 format，
    避免使用者自訂 prompt 裡的 `{...}` 被誤解析）。"""
    tpl = template_path.read_text(encoding='utf-8')
    return tpl.replace('{topic}', topic).replace('{keywords}', ', '.join(terms))


def run_l1(topic, terms, topic_docs, args):
    topic_l1 = TOPICS_DIR / topic / 'L1'
    if args.force and args.layer in ('l1', 'all') and topic_l1.is_dir():
        n = 0
        for f in topic_l1.glob('*/*.md'):
            f.unlink()
            n += 1
        log(f'  --force: 清 L1 {topic} ({n} 份摘要)')
    prompt_text = render_prompt(PROMPT_DIR / 'l1-topic-extract.txt', topic, terms)
    input_budget = args.context_length - args.l1_max_output - 1024
    out_budget = int(args.l1_max_output * 0.85)
    # 預設「不重做」：只把尚未有摘要的篇目送進批次（已產出的 .md 完全不重送）
    pending_docs = [d for d in topic_docs if not l1_path(topic, d).exists()]
    cached = len(topic_docs) - len(pending_docs)
    if cached:
        log(f'L1 [{topic}]: {cached} 篇已有摘要 → 不重做（--layer l1/all --force 才會重做）')
    if not pending_docs:
        log(f'L1 [{topic}]: 全部 {len(topic_docs)} 篇已完成，無需處理')
        return
    batches = build_l1_batches(pending_docs, prompt_text, input_budget, out_budget,
                               args.tokenize_url, max_files=args.l1_max_files)
    log(f'L1 [{topic}]: 待處理 {len(pending_docs)} 篇 → {len(batches)} 個批次')
    kw_str = ', '.join(terms)
    for batch, n_tokens in batches:
        log(f'  L1 [{topic}] 批次: {len(batch)} 篇 ({n_tokens} tokens input)')
        output = llm_call(
            args.api_url, args.model,
            prompt_text + ''.join(doc_section(d) for d in batch),
            args.l1_max_output, temperature=0.1,
            label=f'L1[{topic}:{batch[0]["id"]}]')
        cards = parse_cards(output, batch)
        missing = [d['id'] for d in batch if d['id'] not in cards or not cards[d['id']]]
        if missing:
            log(f'  批次內 {len(missing)} 篇缺摘要: {missing[:5]}{"…" if len(missing) > 5 else ""} → 逐篇重跑')
            for d in [x for x in batch if x['id'] in missing]:
                out2 = llm_call(
                    args.api_url, args.model,
                    prompt_text + doc_section(d),
                    args.l1_max_output, temperature=0.1,
                    label=f'L1-single[{topic}:{d["id"]}]')
                c2 = parse_cards(out2, [d])
                if d['id'] not in c2 or not c2[d['id']]:
                    c2 = {d['id']: out2.strip()}  # 保底：整段視為摘要
                cards[d['id']] = c2[d['id']]
        for d in batch:
            p = l1_path(topic, d)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f'<!-- topic: {topic} | doc: {d["id"]} | {d["date"]} | '
                         f'{d["title"]} | keywords: {kw_str} -->\n'
                         + cards.get(d['id'], '').strip() + '\n',
                         encoding='utf-8')
    log(f'L1 [{topic}] 完成 ({len(pending_docs)} 篇新處理)')


# ---------------------------------------------------------------- L2: reduce

def l2_header(topic, terms, n):
    return (f"Below are {n} keyword-focused summaries (one per document) from the "
            f"2012portal corpus, all about the topic '{topic}' "
            f"(keywords searched: {', '.join(terms)}):\n\n")


def chunk_by_budget(items, input_budget, tokenize_url, max_items=80, header=''):
    """items: [(key, text)]。回傳可放入預算的子塊清單。"""
    chunks, cur, cur_tokens = [], [], 0
    for key, text in items:
        t = estimate_tokens(text) + 20
        if cur and (cur_tokens + t > input_budget or len(cur) >= max_items):
            chunks.append(cur)
            cur, cur_tokens = [], 0
        cur.append((key, text))
        cur_tokens += t
    if cur:
        chunks.append(cur)
    # 精確驗證
    verified = []
    for chunk in chunks:
        body = ''.join(t for _, t in chunk)
        n = tokenize(header + body, tokenize_url)
        if n > input_budget and len(chunk) > 1:
            half = len(chunk) // 2
            log(f'  L2 塊過大 ({n} tokens) → 拆')
            verified.extend(chunk_by_budget(
                chunk[:half], input_budget, tokenize_url, max_items, header)
                + chunk_by_budget(
                    chunk[half:], input_budget, tokenize_url, max_items, header))
        else:
            verified.append(chunk)
    return verified


def merge_parts(prompt_text, parts, args, label):
    """把多份子摘要合併成一份；太大時分組多輪 reduce。"""
    if len(parts) == 1:
        return parts[0]
    input_budget = args.context_length - args.l2_max_output - 1024
    groups, cur, cur_tokens = [], [], 0
    for i, p in enumerate(parts, 1):
        t = estimate_tokens(p) + 20
        if cur and cur_tokens + t > input_budget:
            groups.append(cur)
            cur, cur_tokens = [], 0
        cur.append((i, p))
        cur_tokens += t
    if cur:
        groups.append(cur)
    merged = []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0][1])
        else:
            body = ''.join(f'\n\n===== Topic sub-summary {i} =====\n{p}\n'
                           for i, p in group)
            m = llm_call(
                args.api_url, args.model,
                prompt_text + '\n\n' + body,
                args.l2_max_output, temperature=0.1,
                label=f'{label}-merge({len(group)})')
            merged.append(m.strip())
    if len(merged) > 1:
        return merge_parts(prompt_text, merged, args, label)
    return merged[0]


def run_l2(topic, terms, topic_docs, args):
    out_path = TOPICS_DIR / topic / 'README.md'
    redo = args.force and args.layer in ('l2', 'all')
    if out_path.exists() and not redo:
        log(f'L2 [{topic}]: README.md 已存在 → 略過（--layer l2/all --force 重做）')
        return
    items = []
    for d in topic_docs:
        p = l1_path(topic, d)
        if p.exists():
            items.append((d['id'], p.read_text(encoding='utf-8').strip()))
    if not items:
        log(f'L2 [{topic}]: 尚無任何 L1 摘要 → 略過')
        return
    if len(items) < len(topic_docs):
        log(f'L2 [{topic}]: 只用 {len(items)}/{len(topic_docs)} 份已存在的 L1 摘要')
    prompt_text = render_prompt(PROMPT_DIR / 'l2-topic-merge.txt', topic, terms)
    input_budget = args.context_length - args.l2_max_output - 1024
    header = l2_header(topic, terms, len(items))
    total_est = sum(estimate_tokens(t) for _, t in items)
    log(f'L2 [{topic}]: {len(items)} 份 L1 摘要 (≈{total_est} tokens)')
    chunks = chunk_by_budget(items, input_budget, args.tokenize_url,
                             header=header)
    if len(chunks) == 1:
        part = llm_call(
            args.api_url, args.model,
            prompt_text + header + ''.join(t for _, t in chunks[0]),
            args.l2_max_output, temperature=0.1, label=f'L2-{topic}')
    else:
        log(f'L2 [{topic}]: 分 {len(chunks)} 輪合併')
        parts_dir = TOPICS_DIR / topic / '_parts'
        parts_dir.mkdir(parents=True, exist_ok=True)
        parts = []
        try:
            for i, chunk in enumerate(chunks, 1):
                p = parts_dir / f'part{i}.md'
                part = llm_call(
                    args.api_url, args.model,
                    prompt_text + header + ''.join(t for _, t in chunk),
                    args.l2_max_output, temperature=0.1,
                    label=f'L2-{topic}-part{i}')
                p.write_text(part.strip() + '\n', encoding='utf-8')
                parts.append(part.strip())
            part = merge_parts(prompt_text, parts, args, f'L2-{topic}')
        finally:
            for f in parts_dir.glob('*.md'):
                f.unlink(missing_ok=True)
            try:
                parts_dir.rmdir()
            except OSError:
                pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(part.strip() + '\n', encoding='utf-8')
    log(f'L2 [{topic}] 完成 → {out_path}')


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description='2012portal 主題導向兩階段摘要管線 (EN)')
    ap.add_argument('--layer', choices=['l1', 'l2', 'all'], default='all',
                    help='跑哪個階段（預設 all）')
    ap.add_argument('--force', action='store_true',
                    help='l1/all: 清掉 L1 重做；l2/all: 重跑 L2（依最新 prompt）')
    ap.add_argument('--topics-file', default=str(TOPICS_FILE))
    ap.add_argument('--limit', type=int, default=0,
                    help='只用前 N 篇來源文章 (測試用)')
    ap.add_argument('--api-url', default=DEFAULT_API)
    ap.add_argument('--tokenize-url', default=DEFAULT_TOKENIZE_URL)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--context-length', type=int, default=DEFAULT_CONTEXT)
    ap.add_argument('--l1-max-output', type=int, default=8000)
    ap.add_argument('--l2-max-output', type=int, default=16000)
    ap.add_argument('--l1-max-files', type=int, default=32,
                    help='L1 單批最多幾篇 (2012portal 多為短文章，預設 32)')
    args = ap.parse_args()

    TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    log(f'layer={args.layer} force={args.force} topics-file={args.topics_file} '
        f'api={args.api_url} ctx={args.context_length}')

    docs = discover_docs()
    log(f'來源: {len(docs)} 篇 (去重後)')
    if args.limit:
        docs = docs[:args.limit]
        log(f'--limit {args.limit}: 只用前 {len(docs)} 篇')
    topics = parse_topics(Path(args.topics_file))
    log(f'主題: {len(topics)} 個')

    for topic, terms in topics:
        tdocs = [d for d in docs if any(t in d['hay'] for t in terms)]
        log(f'=== {topic} (關鍵字: {", ".join(terms)}) → {len(tdocs)} 篇相符 ===')
        if not tdocs:
            log(f'    沒有相符文章 → 略過')
            continue
        if args.layer in ('l1', 'all'):
            run_l1(topic, terms, tdocs, args)
        if args.layer in ('l2', 'all'):
            run_l2(topic, terms, tdocs, args)
    log('全部完成')


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as exc:
        print(f'錯誤: {exc}', file=sys.stderr)
        sys.exit(2)
