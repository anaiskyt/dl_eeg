

def save_list(list, file_name):
    file_name += ".csv"
    string = ""
    for k in list:
        string += str(k)
        string += "\n"
    with open(file_name, "w") as f:
        f.write(string)







