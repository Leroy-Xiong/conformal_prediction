from sklearn.datasets import fetch_openml


X, y = fetch_openml(data_id=4353, return_X_y=True, as_frame=False, parser='auto')