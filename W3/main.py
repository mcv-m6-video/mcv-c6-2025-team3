import argparse

# CHANGE PATHS ADAPTING TO YOUR ABSOLUTE PATH:
video_path = r'/Users/andrea.sanchez/Desktop/AICity_data/train/S03/c010/vdo.avi'
annotations_path =  r'/Users/andrea.sanchez/Desktop/ai_challenge_s03_c010-full_annotation.xml'

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Week 3')
    parser.add_argument('--task', type=int, default=1,  help=(
        "Introduce the task number you want to execute:\n"
        "  1.Optical flow off-the-shelf\n"
        "  2.Improve tracking with optical flow\n"
        "  3.Evaluate best tracking algorithm in SEQ01\n"
        "  4.Evaluate best tracking algorithm in SEQ03\n"
    ))
    args = parser.parse_args()

    # python main.py --task 1
    if args.task == 1:
        print("Task 1.1: Optical flow off-the-shelf")
        

    elif args.task == 2:
        print("Task 1.2: Improve tracking with optical flow")
       
    elif args.task == 3:
        print("Task 2.1: Evaluate best tracking algorithm in SEQ01")
        
    elif args.task == 4:
        print("Task 2.2: Evaluate best tracking algorithm in SEQ03")
        
    else:
        print('Task not implemented')
        exit(1)
