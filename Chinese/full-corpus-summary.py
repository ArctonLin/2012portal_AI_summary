#!/usr/bin/env python3
"""full-corpus-summary.py — Cobra 全語料三階段 map-reduce 摘要管線。

對 /rtx-storage/crawler/golden-ages 下 message / interview / meeting 共約 900
篇內容（2012-03 ~ 2026）做「整體摘要」。原文約 24MB（~90k-120k tokens），
超出任何單次 context window，因此採三階段：

  L1 (map)   : 每篇文件 → 一張「事實卡」(fact card)：有日期的事件、實體、
               數字、關鍵宣告。小檔案會批次打包 (≤ input budget) 減少呼叫數。
               逐篇快取 → 可中斷重跑 (resume)。
  L2 (reduce): 依「年」把事實卡合併成年度摘要；單年超預算時先切成子塊
               (multi-round reduce) 再合併。
  L3 (final) : 全部年度摘要 → 一份總體摘要 (含完整時間軸：史前/百萬年前 →
               1996 大入侵 → 2012-2026 → 未來預測)。

所有 token 數以本地 llama.cpp /tokenize 精確計算；任何階段若組裝後的 prompt
超過 context 預算即拆分批次，絕不送出會爆窗的 request。

只讀來源：/rtx-storage/crawler/golden-ages 絕不寫入。

Examples:
    # 小樣本測試 (前 4 篇)
    python3 full-corpus-summary.py --limit 4

    # 只跑某年 (L2 需要該年 L1 完成)
    python3 full-corpus-summary.py --years 2012 --stage l2

    # 完整跑 (會花數小時，可中斷，重跑會跳過已完成項)
    python3 full-corpus-summary.py --stage all
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

BASE_DIR = Path('/storage/workspace/golden-ages-tts')
SOURCE_ROOT = Path('/rtx-storage/crawler/golden-ages')  # 唯讀
CATEGORIES = ['message', 'interview', 'meeting']

DEFAULT_API = os.environ.get('LLAMA_API_URL', 'http://127.0.0.1:8000/v1/chat/completions')
DEFAULT_TOKENIZE_URL = os.environ.get('LLAMA_TOKENIZE_URL', 'http://127.0.0.1:8000/tokenize')
DEFAULT_MODEL = os.environ.get('LLAMA_MODEL', 'qwen')
DEFAULT_CONTEXT = 262144          # llama.cpp n_ctx
CHARS_PER_TOKEN = 1.25            # CJK 混排保守換算 (只用於預估，最終以 /tokenize 為準)

OUT_DIR = BASE_DIR / 'full-corpus'
L1_DIR = OUT_DIR / 'L1'
L2_DIR = OUT_DIR / 'L2'
PROMPT_DIR = BASE_DIR / 'prompts-full-corpus'

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
    """回傳 token 數；端點不可用時 fail closed。"""
    try:
        data = post_json(tokenize_url, {'content': text}, timeout=180)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f'無法使用 /tokenize ({tokenize_url})：{exc}') from exc
    tokens = data.get('tokens')
    if not isinstance(tokens, list):
        raise RuntimeError('/tokenize 回應缺少 tokens 陣列')
    return len(tokens)


def estimate_tokens(text):
    return int(len(text) / CHARS_PER_TOKEN) + 1


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
    """掃描唯讀來源，回傳依 (date, category, folder) 排序的文件清單。
    重複內容 (md5 相同) 只保留第一筆。"""
    docs, seen_md5 = [], set()
    for cat in CATEGORIES:
        cat_dir = SOURCE_ROOT / cat
        if not cat_dir.is_dir():
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
                log(f'  跳過重複內容: {cat}/{folder.name}')
                continue
            seen_md5.add(md5)
            date, title = folder.name.split('__', 1)[:2] if '__' in folder.name else ('?', folder.name)
            meta = folder / 'meta.json'
            if meta.exists():
                try:
                    m = json.loads(meta.read_text(encoding='utf-8'))
                    date = m.get('date', date)
                    title = m.get('title', title)
                except (json.JSONDecodeError, OSError):
                    pass
            doc_id = f'{cat}/{folder.name}'
            docs.append({
                'id': doc_id,
                'cat': cat,
                'folder': folder.name,
                'date': str(date),
                'year': str(date)[:4],
                'title': title,
                'text': text,
            })
    docs.sort(key=lambda d: (d['date'], d['cat'], d['folder']))
    return docs


def doc_section(doc):
    return (f'\n\n===== 檔案 {doc["id"]} =====\n'
            f'標題：{doc["title"]}\n日期：{doc["date"]}\n'
            f'--- 原文 ---\n{doc["text"]}')


# ---------------------------------------------------------------- L1: map

def l1_cache_path(doc):
    p = L1_DIR / doc['cat'] / (doc['folder'] + '.md')
    return p


def build_l1_batches(docs, prompt_text, input_budget, tokenize_url, max_files=8):
    """以字數預估貪心打包；組裝後用 /tokenize 精確驗證，超標即對半拆。"""
    prompt_tokens = tokenize(prompt_text, tokenize_url)
    batches, cur, cur_est = [], [], 0
    for doc in docs:
        est = estimate_tokens(doc['text']) + 80
        if cur and (cur_est + est > input_budget or len(cur) >= max_files):
            batches.append(cur)
            cur, cur_est = [], 0
        cur.append(doc)
        cur_est += est
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
                    batch[:half], prompt_text, input_budget, tokenize_url, max_files)
                    + build_l1_batches(
                        batch[half:], prompt_text, input_budget, tokenize_url, max_files))
                continue
            log(f'  警告: 單篇文件超過 input 預算 ({n} > {input_budget}): {batch[0]["id"]}')
        verified.append((batch, n))
    return verified


CARD_RE = re.compile(r'^##\s*CARD[:：]\s*(\S+)', re.M)


def parse_cards(output, batch):
    """從批次輸出中切出各篇事實卡。回傳 {doc_id: card_text}。"""
    cards = {}
    matches = list(CARD_RE.finditer(output))
    for i, m in enumerate(matches):
        key = m.group(1).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(output)
        cards[key] = output[m.end():end].strip()
    return cards


def run_l1(docs, args):
    prompt_text = (PROMPT_DIR / 'l1-extract.txt').read_text(encoding='utf-8')
    input_budget = args.context_length - args.l1_max_output - 1024
    batches = build_l1_batches(docs, prompt_text, input_budget,
                               args.tokenize_url, max_files=args.l1_max_files)
    log(f'L1: {len(docs)} 篇 → {len(batches)} 個批次')
    todo = 0
    for batch, n_tokens in batches:
        pending = [d for d in batch if not l1_cache_path(d).exists()]
        if not pending:
            log(f'  批次 {batch[0]["id"][:40]}… 已完成 ({len(batch)} 篇)')
            continue
        todo += len(pending)
        log(f'  L1 批次: {len(batch)} 篇 ({n_tokens} tokens input, '
            f'{len(pending)} 篇待處理)')
        output = llm_call(
            args.api_url, args.model,
            prompt_text + ''.join(doc_section(d) for d in batch),
            args.l1_max_output, temperature=0.1,
            label=f'L1[{batch[0]["id"]}]')
        cards = parse_cards(output, batch)
        missing = [d['id'] for d in batch if d['id'] not in cards or not cards[d['id']]]
        if missing:
            log(f'  批次內 {len(missing)} 篇缺卡: {missing} → 逐篇重跑')
            for d in [x for x in batch if x['id'] in missing]:
                out2 = llm_call(
                    args.api_url, args.model,
                    prompt_text + doc_section(d),
                    args.l1_max_output, temperature=0.1,
                    label=f'L1-single[{d["id"]}]')
                c2 = parse_cards(out2, [d])
                if d['id'] not in c2 or not c2[d['id']]:
                    c2 = {d['id']: out2.strip()}  # 保底：整段視為卡
                cards[d['id']] = c2[d['id']]
        for d in batch:
            p = l1_cache_path(d)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f'<!-- card: {d["id"]} | {d["date"]} | {d["title"]} -->\n'
                         + cards.get(d['id'], '').strip() + '\n',
                         encoding='utf-8')
    log(f'L1 完成 ({todo} 篇新處理)')


def load_cards(docs):
    """載入 L1 事實卡 (含 meta header)。"""
    cards = []
    for d in docs:
        p = l1_cache_path(d)
        if not p.exists():
            raise RuntimeError(f'L1 缺失: {p}')
        cards.append({'doc': d, 'text': p.read_text(encoding='utf-8').strip()})
    return cards


# ---------------------------------------------------------------- L2: year reduce

def chunk_by_budget(items, input_budget, tokenize_url, max_items=60):
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
        n = tokenize(header_text(len(chunk)) + body, tokenize_url)
        if n > input_budget and len(chunk) > 1:
            half = len(chunk) // 2
            log(f'  L2 塊過大 ({n} tokens) → 拆')
            verified.extend(chunk_by_budget(
                chunk[:half], input_budget, tokenize_url, max_items)
                + chunk_by_budget(
                    chunk[half:], input_budget, tokenize_url, max_items))
        else:
            verified.append(chunk)
    return verified


def header_text(n):
    return f'以下是 {n} 張事實卡（同一年的 Cobra 揭露文件）：\n\n'


def run_l2(years, docs, args):
    prompt_text = (PROMPT_DIR / 'l2-year-merge.txt').read_text(encoding='utf-8')
    input_budget = args.context_length - args.l2_max_output - 1024
    for year in years:
        out_path = L2_DIR / f'{year}.md'
        if out_path.exists() and not args.force:
            log(f'L2 {year}: 已完成 → 跳過')
            continue
        cards = [c for c in load_cards(docs) if c['doc']['year'] == year]
        if not cards:
            log(f'L2 {year}: 沒有文件')
            continue
        total_est = sum(estimate_tokens(c['text']) for c in cards)
        log(f'L2 {year}: {len(cards)} 張卡 (≈{total_est} tokens)')
        items = [(c['doc']['id'], c['text']) for c in cards]
        chunks = chunk_by_budget(items, input_budget, args.tokenize_url)
        if len(chunks) == 1:
            part = llm_call(
                args.api_url, args.model,
                prompt_text + header_text(len(chunks[0]))
                + ''.join(t for _, t in chunks[0]),
                args.l2_max_output, temperature=0.1, label=f'L2-{year}')
        else:
            log(f'L2 {year}: 分 {len(chunks)} 輪合併')
            parts = []
            for i, chunk in enumerate(chunks, 1):
                p = L2_DIR / f'{year}.part{i}.md'
                if p.exists() and not args.force:
                    log(f'  part {i} 已完成 → 跳過')
                    parts.append(p.read_text(encoding='utf-8').strip())
                    continue
                part = llm_call(
                    args.api_url, args.model,
                    prompt_text + header_text(len(chunk))
                    + ''.join(t for _, t in chunk),
                    args.l2_max_output, temperature=0.1,
                    label=f'L2-{year}-part{i}')
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(part.strip() + '\n', encoding='utf-8')
                parts.append(part.strip())
            merge_input = prompt_text + '\n\n' + ''.join(
                f'\n\n===== 年度子摘要 {i} =====\n{p}\n' for i, p in enumerate(parts, 1))
            part = llm_call(
                args.api_url, args.model, merge_input,
                args.l2_max_output, temperature=0.1,
                label=f'L2-{year}-merge')
            for i in range(1, len(chunks) + 1):
                (L2_DIR / f'{year}.part{i}.md').unlink(missing_ok=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(part.strip() + '\n', encoding='utf-8')
        log(f'L2 {year} 完成 → {out_path}')


# ---------------------------------------------------------------- L3: final

def run_l3(years, docs, args):
    prompt_text = (PROMPT_DIR / 'l3-master.txt').read_text(encoding='utf-8')
    input_budget = args.context_length - args.l3_max_output - 1024
    year_files = []
    for year in years:
        p = L2_DIR / f'{year}.md'
        if not p.exists():
            raise RuntimeError(f'L3 缺少年度摘要: {p}（先跑 --stage l2）')
        year_files.append((year, p.read_text(encoding='utf-8').strip()))
    body = ''.join(f'\n\n===== 年度摘要 {y} =====\n{t}\n' for y, t in year_files)
    full = prompt_text + body
    n = tokenize(full, args.tokenize_url)
    log(f'L3: {len(year_files)} 份年度摘要, input {n} tokens')
    if n > input_budget:
        raise RuntimeError(
            f'L3 input {n} tokens 超過預算 {input_budget}；'
            f'請縮短 prompt 或增大 context')
    out = llm_call(args.api_url, args.model, full,
                   args.l3_max_output, temperature=0.2, label='L3')
    out_path = OUT_DIR / 'full-corpus-summary.md'
    out_path.write_text(out.strip() + '\n', encoding='utf-8')
    log(f'L3 完成 → {out_path}')


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description='Cobra 全語料三階段摘要管線')
    ap.add_argument('--stage', choices=['l1', 'l2', 'l3', 'all'], default='all')
    ap.add_argument('--limit', type=int, default=0,
                    help='只用前 N 篇 (測試用)')
    ap.add_argument('--years', default='',
                    help='逗號分隔年份 (如 2012,2013)；預設全部有文件的年')
    ap.add_argument('--api-url', default=DEFAULT_API)
    ap.add_argument('--tokenize-url', default=DEFAULT_TOKENIZE_URL)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--context-length', type=int, default=DEFAULT_CONTEXT)
    ap.add_argument('--l1-max-output', type=int, default=12000)
    ap.add_argument('--l2-max-output', type=int, default=12000)
    ap.add_argument('--l3-max-output', type=int, default=20000)
    ap.add_argument('--l1-max-files', type=int, default=8,
                    help='L1 單批最多幾篇')
    ap.add_argument('--force', action='store_true',
                    help='重做已存在的 L2/L3 輸出')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f'stage={args.stage} limit={args.limit} api={args.api_url} '
        f'ctx={args.context_length}')

    docs = discover_docs()
    log(f'來源: {len(docs)} 篇 (去重後), 年份 '
        f'{min(d["year"] for d in docs)}~{max(d["year"] for d in docs)}')
    if args.limit:
        docs = docs[:args.limit]
        log(f'--limit {args.limit}: 只用前 {len(docs)} 篇')
    if args.years:
        wanted = set(y.strip() for y in args.years.split(',') if y.strip())
    else:
        wanted = sorted({d['year'] for d in docs})

    if args.stage in ('l1', 'all'):
        run_l1(docs, args)
    if args.stage in ('l2', 'all'):
        run_l2([y for y in wanted], docs, args)
    if args.stage in ('l3', 'all'):
        run_l3([y for y in wanted], docs, args)
    log('全部完成')


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as exc:
        print(f'錯誤: {exc}', file=sys.stderr)
        sys.exit(2)
