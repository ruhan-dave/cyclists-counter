This folder will be used to store all *.csv files and the respective TensorFlow *.record files, which contain the list of annotations for our dataset images.

To have the corresponding .record files here, you can:
1. Run the notebook generate_tfrecords
2. Download them:
	* Download the data:
	```
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1iEG6157WrYrWy5l_9vLQT9i93EOFYQot' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1iEG6157WrYrWy5l_9vLQT9i93EOFYQot" -O tfrecords.zip && rm -rf /tmp/cookies.txt
	```
	The data can also be download manually [here](https://drive.google.com/file/d/1iEG6157WrYrWy5l_9vLQT9i93EOFYQot/view?usp=sharing).

	* Unzip, delete the zip and move the files to the current directory:
	```
	unzip tfrecords.zip
	rm tfrecords.zip
	mv tfrecords/* .
	rmdir tfrecords
	```
