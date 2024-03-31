
import pickle
import pandas as pd

#Load model
with open('model-penguin-66130700355.pkl', 'rb') as file:
    # Load the data from the file
    model, species_encoder, island_encoder ,sex_encoder = pickle.load(file)

#Get New data
x_new = pd.DataFrame()
# Get user input for each variable
x_new['island'] = [input('Enter island (Torgersen, Biscoe, Dream): ')]
x_new['culmen_length_mm'] = [float(input('Enter culmen length in mm (0 to 60): '))]
x_new['culmen_depth_mm'] = [float(input('Enter culmen depth in mm (0 to 60): '))]
x_new['flipper_length_mm'] = [float(input('Enter flipper length in mm (0 to 200): '))]
x_new['body_mass_g'] = [float(input('Enter body mass in g (1000 to 5000): '))]
x_new['sex'] = [input('Enter sex (MALE or FEMALE): ')]

#Encoding
x_new['island'] = island_encoder.transform(x_new['island'])
x_new['sex'] = sex_encoder.transform(x_new['sex'])

#Prediction
y_pred_new = model.predict(x_new)
result = species_encoder.inverse_transform(y_pred_new) 
print('Predicted Specie: ', result)
