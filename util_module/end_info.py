from datetime import datetime

def show_info(out_dir,exec_model):
    #print("result: "+out_dir)
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print((f"End Time : {formatted_time}"))
    #recordInfo(out_dir,exec_model.conditions, formatted_time)
    
def recordInfo(out_dir, info, formatted_time):
    file_path = out_dir + "/info.txt"
    with open(file_path, "w") as file:
        file.write(f"End Time : {formatted_time}\n")
        for x in info:
            file.write(x[0]+" : "+str(x[1])+"\n")