from svmutil import *
from csv2libsvm import *
import csv

train_data_file = csv2libsvm('datatest1.csv', 5, True)
test_data_file = csv2libsvm('datatest2.csv', 5, True)
present_str = "present"
absent_str = "absent"

# labels, items = svm_read_problem(<input file in libsvm format>)
x, y = svm_read_problem(train_data_file)
a, b = svm_read_problem(test_data_file)
# Eg: a, b = [1,1,1], [[1, 0, 1],[-1,0,-1], [-1, 10, 222]]

# Accuracy values for various combinations of kernel, C and datasets
# (Training dataset, Test dataset) = Accuracy
# LINEAR C=30 (3,2) = 98.2055; (1,3) = 91.40; (2,3) = 98.80; (1_2,3) = 98.14
#             (2_3, 1) = 97.82; (1_3, 2) = 99.09 (1_3, rand) = 97.9
#            (2, ran) = 97.8; (1_2_3, rand) = 97.9; (2_3, ran) = 97.8
#       C=31 (1_2,3) = 98.39 (2_3,1) = 96.21 
# POLY C=8.2 (3,2) = 81.72; (2,1) = 97.78 (1,3) = 92.68 (1_3, rand) = 68.6

prob = svm_problem(x,y)
param = svm_parameter()
param.kernel_type = 0
param.C = 30
m = svm_train(prob, param)

# svm_save_model('heart_scal.model', m)
# m = svm_load_model('heart_scal.model')

print param.kernel_type
print param.C

# computed labels will be in p_labs as 1.0 or 0.0
p_labs, p_acc, p_vals = svm_predict(a,b,m)
#print p_labs

# Create result file and write predicted results to it
with open("result.csv", "wb") as csv_file:
    writer = csv.writer(csv_file, lineterminator='\n')
    for item in p_labs:
        if item == 1.0:
            writer.writerow([present_str])
        else:
            writer.writerow([absent_str])
