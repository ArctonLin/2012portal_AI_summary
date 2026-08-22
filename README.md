# Introduction

## Summary runner:
I use Local LLM to do this summmary task. Didn't connect to internet. Not censored by cloud AI.

LLM Hardware: NVIDIA GB10 128GB/2TB + 108TB NAS Storage with 10Gbps Ethernet

LLM Software: llama-b10549-cuda-arm64 (sm_121)

LLM Model: Qwen3.8-27B-NVFP4-MTP-Q8attn

LLM Setting: MTP=2 / ContentWindow=262144

## Code:
Our code is generate by above Local LLM and run full night to abstract and summary 910 post/meeting/interview by Layer 1, L2, L3

L1: Single post summary
L2: Entire year summary
L3: All year summary

## Source 
Source data from 2012portal.blogspot.com and www.golden-ages.org

English Summary here: [full-corpus-summary.md](English/full-corpus-summary.md)

Chinese Summary here: [full-corpus-summary.md](Chinese/full-corpus-summary.md)

## Notice
AI may be wrong.

The text that AI read is based on [Cobra Voice EN](https://cobra-voice.net/en/) and [Cobra Voice CH](https://cobra-voice.net)

You can use [English Search Site](https://cobra-voice.net/en/advanced_search.php) to scan English original post.

Or can use [Chinese Search Site](https://cobra-voice.net/advanced_search.php) to scan Chinese original post.

## Future work
The English version only contain 2012portal post, the meeting and interview is not included.

Only Chinese version contain all post, meeting and interview.

But the AI is not good at chinese analysis.

So I will collect the English version full data then try again if I have spare time.

Feel free to open a issue in github here if you have any suggestion.

Victory of the Light!