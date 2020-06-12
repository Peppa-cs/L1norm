import numpy as np
import onnxruntime
from timeit import default_timer as timer
import sys,getopt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    return softmax(np.array(result)).tolist()

dict = unpickle('/home/peppa3/data/cifar-10/test_batch')

data = dict[b'data']
labels = dict[b'labels']  # 0~9
data = data.reshape(10000, 3, 32, 32).astype("float32")
labels = np.array(labels)

iteration = 1500

#mean = np.array([0.4914, 0.4822, 0.4465])
#std  = np.array([0.2023, 0.1994, 0.2010])



try:
   opts,args = getopt.getopt(sys.argv[1:],"hi:o:b:",["ifile=","ifile=","batchsize="])
except getopt.GetoptError:
   print('pythoncifar10.py -i <ifile> -o <ofile> -b <batchsize>')
   sys.exit(2)

for opt,arg in opts:
   if opt == '-h':
      print('pythoncifar10.py -i <ifile> -o <ofile> -b <batchsize>')
      sys.exit()
   elif opt in ('-i','--ifile'):
      inputfile = arg
   elif opt in ('-o','--ofile'):
      outputfile = arg
   elif opt in ('-b','--batchsize'):
      batchsize = arg

print('input file is ',inputfile)
print('output file is ',outputfile)
print('batchsize is ',batchsize)



session = onnxruntime.InferenceSession(inputfile, None)
session.set_providers(['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name  
"""
input = np.zeros((8,3,32,32)).astype("float32")
raw_result = session.run([], {input_name: input})
#res = postprocess(raw_result)
print(raw_result)
print(np.argmax(raw_result,axis=2).reshape(-1))

res = postprocess(raw_result)

idx = np.argmax(res)

sort_idx = np.flip(np.squeeze(np.argsort(res)))

if idx == labels[i*batchsize:(i+1)*batchsize]:
   success = success +1
"""
res = open(outputfile,'a')
success = 0


res.write('- batchsize = '+str(batchsize)+' \n')

mean = np.arange(int(batchsize)*3*32*32)
mean = mean.reshape(int(batchsize), 3, 32, 32).astype("float32")
std = np.arange(int(batchsize)*3*32*32)
std = std.reshape(int(batchsize), 3, 32, 32).astype("float32")

for i in range(0,int(batchsize),1):
    mean[i][0] = 0.4914
    mean[i][1] =  0.4822
    mean[i][2] = 0.4465
    std[i][0] = 0.2023
    std[i][1] = 0.1994
    std[i][2] = 0.2010

j = 0

for i in range (0,iteration,1):

  if (j+1)*int(batchsize) > 10000 :
     j=0
  start = timer()
  input = (data[j*int(batchsize):(j+1)*int(batchsize)] /255.0 -mean) / std
  input = input.astype("float32")
  raw_result = session.run([], {input_name: input})
  idx = np.argmax(raw_result,axis=2).reshape(-1)
  success += sum(idx == labels[j*int(batchsize):(j+1)*int(batchsize)])
  end = timer()
  res.write('one batch time is: '+str(end-start)+' s\n')
  j += 1

res.close()
print('success = ' + str(success))


