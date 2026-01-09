#CONNECT TO RASPBERRYPI VIA SSH
-> ssh wisdom@**.**.**.***
->password:  WISDOM
-----------------------------------------------------------------------------------------------------------
Access Dashboard with the link Below

-> https://rajeshs1719.github.io/wisdom/

-----------------------------------------------------------------------------------------------------------
To run Waste classifier with model.

-> cd waste_sorter
-> source tflite_env/bin/activate
-> python3 classifier_app.py

----------------------------------------------------------------------------------------------------------
To run Waste classifier with Gemini.

-> cd waste_sorter
-> source tflite_env/bin/activate
-> export GEMINI_API_KEY="Your Gemini API Key"
-> python3 waste_sort_oled.py

-----------------------------------------------------------------------------------------------------------