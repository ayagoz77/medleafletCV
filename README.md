# medleafletCV
repo for detecting text and processing prompts related with detected text with llama3 api
-
Install requirements into your env.
```bash
pip install -r requirements.txt
```
Create .env file and set variables GROQ_API_KEY(groq cloud api key) and TOKEN(bot father from telegram)
Finally simply run
```bash
python tg_bot.py
```
-
Use text_handler.py with your cfg.yaml
```bash
python text_handler.py --cfg your_cfg.yaml
```
Each folder explanation for further unit testing:
```tree
├── block_extractor               
│   ├── block_id_y.py           - need to improve (block clutering after getting columns) not used
│   └── density_clustering_x.py - clustering into by x values
├── detectors
│   ├── corner_detector.py      - one point detector
│   ├── detector.py             - abstract class
│   ├── document_detector.py    - 4 point detector
│   ├── __init__.py
│   ├── model_weights           - passed as default
│   │   ├── cornerDetector.onnx
│   │   └── docDetector.onnx
│   └── test
│       └── quality_check.py    - calculates mean IoU (needs input as image.jpg and its label as image.jpg.csv with 4 points coordinates(order doesnt matter))
└── utils
    ├── __init__.py
    ├── quality_enhancer.py     - brightness and contrast enhancer
    └── text_extractor.py       - morphology operator for text bboxes extracting
```
