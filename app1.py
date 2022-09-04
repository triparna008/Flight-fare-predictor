import streamlit as st
import pickle
import numpy as np
import pandas as pd
import datetime

def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://vfltravelservices.com/wp-content/uploads/2021/03/flight.jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_bg_hack_url()

model = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title('Flight Fare Predictor')
st.sidebar.image('data:image/jpeg;base64,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')
airline = st.sidebar.selectbox('Airline', df['Airline'].unique())
source = st.sidebar.selectbox('Source', df['Source'].unique())
destination = st.sidebar.selectbox('Destination', df['Destination'].unique())
journey = st.date_input('Date of Journey')
departure = st.time_input('Departure Time')
arrival = st.time_input('Arrival Time')
stops = st.selectbox('Total Stops', [0,1,2,3,4])



if st.button('Predict Fare'):
    journey_day = int(journey.day)
    journey_month = int(journey.month)
    
    dep_hour = int(departure.hour)
    dep_minute = int(departure.minute)
    
    arr_hour = int(arrival.hour)
    arr_minute = int(arrival.minute)
    
    dep_time = dep_hour*60 + dep_minute
    arr_time = arr_hour*60 + arr_minute
    
    if arr_time > dep_time:
        total_duration = arr_time - dep_time
        duration_hour = total_duration//60
        duration_minute = total_duration%60
    else:
        total_duration = (1440 - dep_time + arr_time)
        duration_hour = total_duration//60
        duration_minute = total_duration%60 
  
    
#     if arr_hour > dep_hour:
#         duration_hour = arr_hour - dep_hour
#     else:
#         duration_hour = (24 - dep_hour + arr_hour)
#     if arr_minute > dep_minute:
#         duration_minute = arr_minute - dep_minute
#     else:
#         duration_minute = (60 - dep_minute + arr_minute)
    
    
    if(airline=='Jet Airways'):
        Jet_Airways = 1
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0 

    elif (airline=='IndiGo'):
        Jet_Airways = 0
        IndiGo = 1
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0 

    elif (airline=='Air India'):
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 1
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0 
            
    elif (airline=='Multiple carriers'):
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 1
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0 
            
    elif (airline=='SpiceJet'):
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 1
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0 
            
    elif (airline=='Vistara'):
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 1
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0

    elif (airline=='GoAir'):
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 1
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0

    elif (airline=='Multiple carriers Premium economy'):
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 1
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0

    elif (airline=='Jet Airways Business'):
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 1
        Vistara_Premium_economy = 0
        Trujet = 0

    elif (airline=='Vistara Premium economy'):
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 1
        Trujet = 0
            
    elif (airline=='Trujet'):
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 1

    else:
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0


        # Source
    if (source == 'Delhi'):
        Delhi = 1
        Kolkata = 0
        Mumbai = 0
        Chennai = 0
        Bangalore = 0

    elif (source == 'Kolkata'):
        Delhi = 0
        Kolkata = 1
        Mumbai = 0
        Chennai = 0
        Bangalore = 0

    elif (source == 'Mumbai'):
        Delhi = 0
        Kolkata = 0
        Mumbai = 1
        Chennai = 0
        Bangalore = 0

    elif (source == 'Chennai'):
        Delhi = 0
        Kolkata = 0
        Mumbai = 0
        Chennai = 1
        Bangalore = 0

    else:
        Delhi = 0
        Kolkata = 0
        Mumbai = 0
        Chennai = 0
        Bangalore = 1


        # Destination
    if (destination == 'Cochin'):
        Cochin = 1
        Delhi = 0
        New_Delhi = 0
        Hyderabad = 0
        Kolkata = 0
        Bangalore = 0
        
    elif (destination == 'Delhi'):
        Cochin = 0
        Delhi = 1
        New_Delhi = 0
        Hyderabad = 0
        Kolkata = 0
        Bangalore = 0

    elif (destination == 'New_Delhi'):
        Cochin = 0
        Delhi = 0
        New_Delhi = 1
        Hyderabad = 0
        Kolkata = 0
        Bangalore = 0

    elif (destination == 'Hyderabad'):
        d_Cochin = 0
        d_Delhi = 0
        d_New_Delhi = 0
        d_Hyderabad = 1
        d_Kolkata = 0
        Bangalore = 0

    elif (destination == 'Kolkata'):
        Cochin = 0
        Delhi = 0
        New_Delhi = 0
        Hyderabad = 0
        Kolkata = 1
        Bangalore = 0

    else:
        Cochin = 0
        Delhi = 0
        New_Delhi = 0
        Hyderabad = 0
        Kolkata = 0
        Bangalore = 1
        
    query = np.array([[airline, source, destination, stops, journey_day, journey_month, dep_hour, dep_minute, arr_hour, arr_minute, duration_hour, duration_minute]])
    prediction = np.round(model.predict(query)[0])
    st.text('Total duration is ' + str(duration_hour) + ' hour ' + str(duration_minute) + ' minute')
    st.header('The predicted fare is around Rs. ' + str(prediction))