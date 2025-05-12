Run general Data to get statistic 


Run createExcel to generate excel file in the right format
default output.csv you can modify and use this file in the main.py
Should use the original data to create the right format data for using later.

Run main.py
Example run on sample.xlsx 


statistic error, and predicted concentration


### Run backend application
move to the backend directory:
```
 cd backend
```

### Create a Virtual Environment
```
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### Install Dependencies
```
    pip install -r backend/requirements.txt
```

### Run the API Server
```
    uvicorn app:app --reload
```

## 📬 API Usage
POST /simulate
Request Body (JSON):

```
{
  "target_cmin": 10.0,
  "new_patient": [
    {
      "patient": 999,
      "time": 0,
      "amt": 1000,
      "evid": 1,
      "conc": null,
      "weight": 75,
      "scr": 1.1,
      "age": 55,
      "gender": 0
    },
    ...
  ]
}

```