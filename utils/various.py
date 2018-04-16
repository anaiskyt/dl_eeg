# Various utils

def save_list(list, file_name):
    file_name += ".csv"
    string = ""
    for k in list:
        string += str(k)
        string += "\n"
    with open(file_name, "w") as f:
        f.write(string)

def save2Dmat(mat, file_name):
    file_name += ".csv"
    string = ""
    for line in mat:
        for i in line:
            string += str(i)
            string += ","
        string += "\n"
    with open(file_name, "w") as f:
        f.write(string)










