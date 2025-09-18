import os
from glob import glob


def cpdata(dst_dir):
    if os.path.exists(dst_dir):
        os.system(f"rm -r {dst_dir}")
    
    os.makedirs(dst_dir, exist_ok=True)

    filed_dir = []
    for IDDD in algo_id:
        file_dir = os.path.join(cvat_dir, str(IDDD))
        file_save_dir = os.path.join(dst_dir, str(IDDD))

        if os.path.exists(file_save_dir):
            continue

        try:
            os.system(f"sudo cp -r {file_dir} {file_save_dir}")
            command1 = f"""
                            cd {file_save_dir} &&
                            sudo unzip *.zip && 
                            sudo rm -r *.zip
                        """

            os.system(command1)
        except:
            filed_dir.append(IDDD)
            continue
    os.system(f"sudo chmod -R 777 {dst_dir}")

    print(filed_dir)

def clear_other(data_dir, dst_dir2, target_parts, clear_dir):
    if os.path.exists(dst_dir2):
        os.system(f"rm -r {dst_dir2}")

    os.makedirs(dst_dir2, exist_ok=True)
    subdirs = os.listdir(data_dir)
    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)

        metainfos_path = os.path.join(subdir_path, "metainfo")
        mv = False
        for csv_path in glob(os.path.join(metainfos_path, "*")):
            for target_part in target_parts:
                if target_part in csv_path:
                    mv = True
                    break
            if mv:
                break
        if mv:
            command1 = f"""
                cp -r {subdir_path} {dst_dir2} &&
                rm -r {subdir_path}
            """
            os.system(command1)
        else:
            continue

        for filedir in clear_dir:
            clear_path = os.path.join(dst_dir2, subdir, filedir)
            if os.path.exists(clear_path):
                os.system(f"rm -r {clear_path}")
    
    dst_dir2_name = os.path.basename(dst_dir2)
    os.system(f"cd /home/xianjianming/data && zip {dst_dir2_name}.zip -r {dst_dir2_name}")
    os.system(f"rm -r {dst_dir2}")


def clear_data(data_dir):
    subdirs = os.listdir(data_dir)
    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        data_path = os.path.join(subdir_path, "data")
        data2_path = os.path.join(subdir_path, "data2")
        os.makedirs(data2_path, exist_ok=True)

        singlepinpad_path = os.path.join(subdir_path, "singlepinpad")

        for xxxdir in glob(os.path.join(singlepinpad_path, "*")):
            for filename in os.listdir(xxxdir):
                uuid, fiex = filename.split("#")
                newfiex = fiex.split("_")[:2]
                xxxxx = [uuid] + newfiex
                new_name = "_".join(xxxxx)

                data_img = new_name+".png"
                data_txt = new_name+".txt"

                data_img_path = os.path.join(data_path, data_img)
                data_txt_path = os.path.join(data_path, data_txt)

                os.system(f"cp {data_img_path} {data2_path}")
                os.system(f"cp {data_txt_path} {data2_path}")
                pass
        
        os.system(f"rm -r {data_path}")
        os.system(f"mv {data2_path} {data_path}")

def count_imgs(data_dir):
    subdirs = os.listdir(data_dir)
    nums = 0
    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        data_path = os.path.join(subdir_path, "data")

        nums += len(os.listdir(data_path))//2
    print(nums)

algo_id = [
47245,47252
]

clear_dir = ["padgroup", "body", "component"]
date = '250814'
DA = '1619_fzqzzl_xh_op'
# SMT = ''
cvat_dir = "/cvat-sync/defect/"
dst_dir = "/home/xianjianming/data/from_cvat/"
cpdata(dst_dir)
dst_dir2 = f"/home/xianjianming/data/defect_DA{DA}_{date}"
clear_other(dst_dir, dst_dir2, [ "singlepad",  "singlepinpad"], clear_dir)
# count_imgs(dst_dir)
    
# clear_data(dst_dir2)
# os.system(f"zip 20240910.zip -r {dst_dir}")