import pyrebase

firebaseConfig = {
    "apiKey": "AIzaSyB-J9gSyV5wK7qomTZucwfYdI9HjlSFOuQ",
    "authDomain": "csi03-spck.firebaseapp.com",
    "databaseURL": "https://csi03-spck-default-rtdb.firebaseio.com",
    "projectId": "csi03-spck",
    "storageBucket": "csi03-spck.appspot.com",  
    "messagingSenderId": "724495220738",
    "appId": "1:724495220738:web:819db1d032662f542de71b",
    "measurementId": "G-M9QGVVDY5G"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
