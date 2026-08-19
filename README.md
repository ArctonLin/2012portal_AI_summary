# Introduction

## Summary runner:
我使用本地 LLM 執行摘要任務，全程不連網，所有能量影響的隨機函數都是在本地端運作。

LLM Hardware: NVIDIA GB10 128GB/2TB + 108TB NAS Storage with 10Gbps Ethernet

LLM Software: llama-b10327-vulkan-arm64

LLM Model: Qwen3.8-27B-NVFP4-MTP-Q8attn

## Code:

程式碼是上述本地 LLM 產生的摘要程式碼，並且運行了一個晚上才將 910 篇中文內容摘要，並分層 L1 (單篇), L2 (年份), L3(全部)

程式碼目前尚不公開

## 原始資料

AI產出的原始內容在 [full-corpus-summary.md](Chinese/full-corpus-summary.md)

只是它對於事件發生的時間點太過自信了，加上部分內容並未完全妥善參考

例如實體頂夸克炸彈拆完後還有電漿頂夸克炸彈... 這部分因為跨年份所以無法妥善摘要

原因是 LLM 一次輸入的資料量有限制，不能將全篇幅資料一次輸入，所以造成摘要錯誤

## 聲明啟事

由於 LLM 只是協助我們整理大致的時間軸

所以以下內容為 AI 產出的結過，並非 Cobra 說過的原話

詳細內容請使用 [搜尋網站](https://cobra-voice.net/advanced_search.php) 進行檢索查證

## 人工查證

預計未來會透過人工查證修正以下內容 (但還沒有時間)