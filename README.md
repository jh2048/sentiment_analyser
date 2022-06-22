# Net Purpose Machine Learning Coding Challenge

### Subtasks
- Explore the data and come up with a visualization of which products have the most positive and negative tweets
> Find in notebooks/EDA.ipynb

- Train a classifier to determine the sentiment of the tweets (Positive/Negative/Neutral)
> Find in notebooks/modelling.ipynb

- "deploy" the classifier using a docker container and an endpoint of "/predict" to run predictions on new tweets
  - I'd recommend using flask as a simple server
> Instructions below

# Build & Run Container
```bash
docker image build -t sentiment_prediction prediction/

docker run -it -p 5000:5000 -d sentiment_prediction:latest 
```

### Invoked via URL
Add sentence and model to URL. Results then displayed.
```
http://localhost:5000/predict_page/<sentence>/<model>
```
### Invoked via Script (JSON)
Use request_sentiment.py
