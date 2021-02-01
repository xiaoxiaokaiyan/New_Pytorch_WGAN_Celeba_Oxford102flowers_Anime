from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=256,height=256):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)   
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)
for jpgfile in glob.glob("C:\\Users\\33203\\Desktop\\WGAN\\images\\aa\\*.jpg"):
    convertjpg(jpgfile,"C:\\Users\\33203\\Desktop\\WGAN\\images\\images")