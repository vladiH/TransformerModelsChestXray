# Download the 56 zip files in Images_png in batches
import urllib.request
import os
import tarfile

def unzip_files(source_directory: str, destination_directory: str):
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_directory, exist_ok=True)
    
    # Iterate through each file in the source directory
    for file_name in os.listdir(source_directory):
        file_path = os.path.join(source_directory, file_name)
        
        # Check if the file is a .tar.gz file
        if file_name.endswith(".tar.gz"):
            # Extract the contents of the .tar.gz file
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(destination_directory)
            
            os.remove(file_path)
            print("Extraction successful:", file_name)
        else:
            print("Skipping non-.tar.gz file:", file_name)
    print("Unzip files done")

def download_all_images(source_directory):
    # Create the destination directory if it doesn't exist
    os.makedirs(source_directory, exist_ok=True)
    # URLs for the zip files
    links = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
    ]

    for idx, link in enumerate(links):
        fn = 'images_%02d.tar.gz' % (idx+1)
        download_path = os.path.join(source_directory, fn)
        print ('downloading', fn, '...')
        urllib.request.urlretrieve(link, download_path)  # download the zip file
    print ("Download complete. Please check the checksums")

if __name__ == '__main__':
    source_directory = "../../data/"
    destination_directory = "../../data/"
    download_all_images(source_directory)
    unzip_files(source_directory, destination_directory)
