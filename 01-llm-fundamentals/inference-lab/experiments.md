# InferenceLab Experiments

## Temperature simulation

Logits:
[4.0, 2.0, 1.0]

### Temperature 0.5
- Paris: 0.9796
- London: 0.0179
- banana: 0.0024

### Temperature 1.0
- Paris: 0.8438
- London: 0.1142
- banana: 0.0420

### Temperature 2.0
- Paris: 0.6285
- London: 0.2312
- banana: 0.1402

## Observations

- Lower temperature made the probability distribution sharper.
- Higher temperature gave lower-probability tokens more probability.
- Temperature changes sampling behavior, not the model's knowledge.

## Gemini repeated-prompt experiment
the prompt: Complete this sentence in one short phrase:The robot opened the mysterious box and found
### Run 1
Response: a single, ticking clock.
Input tokens:17
Output tokens:6
Latency:5085.71 ms

### Run 2
Response: a single glowing gear.
Input tokens:17
Output tokens:5
Latency:1577.54 ms

### Run 3
Response: a single, beating heart.
Input tokens:17
Output tokens:6
Latency:4866.26 ms

## Conclusions

- Identical prompts should have the same input token count.
- Generated outputs may differ because generation involves sampling.
- Output token counts may therefore differ.
- API latency can differ even for identical requests.