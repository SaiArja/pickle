import pickle
model = pickle.load(open('diab_79.pkl', 'rb'))
output = model.predict([[1,2,3,4,5,6,7,8]])
if output[0]==1:
    print('diabetic')
else:
    print('not diabatic')