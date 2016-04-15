from svmutil import *
from csv2libsvm import *
import csv

#Pass False as last parameter in-case of no header in the the test file.
test_data_file = csv2libsvmtest('test.csv', True)
#test_data_file = csv2libsvm('random.csv', 5, True)

present_str = "present"
absent_str = "absent"

a, b = svm_read_problem(test_data_file)

m = svm_load_model('iot_a4.model')


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
