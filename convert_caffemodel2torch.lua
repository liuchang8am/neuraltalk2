require 'loadcaffe'
require 'nn'
require 'inn'
require 'cudnn'

-- #TODO add backend option to also support for cpu mode

prototxt_name = '/home/lc/Caffe/caffemodels/fast-rcnn/test_flickr8k.prototxt'
binary_name = '/home/lc/Caffe/caffemodels/fast-rcnn/vgg16_fast_rcnn_iter_40000_flickr8k.caffemodel'

local num_classes = 213

--prototxt_name = '/home/lc/Caffe/caffemodels/vgg16/VGG_ILSVRC_16_layers_deploy.prototxt'
--binary_name = '/home/lc/Caffe/caffemodels/vgg16/VGG_ILSVRC_16_layers.caffemodel'

params = loadcaffe.load(prototxt_name, binary_name, 'cudnn' ) -- some layers will fail
                                                             -- i.e.
                                                             -- warning: module 'data [type Python]' not found
                                                             -- warning: module 'roi_pool5 [type ROIPooling]' not found -- replace with inn.ROIPooling
                                                             -- warning: module 'fc7_drop7_0_split [type Split]' not found
print ("model loaded:", params)
params = params:parameters() -- get params
print ("model parameters:", params) 

-- torch.save('/home/lc/Caffe/caffemodels/fast-rcnn/frcnn_vgg16.t7', params) -- not now, let's save later when ROIPooling is added

local features = nn.Sequential()
local classifier = nn.Sequential()

-- features
features:add(cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, 1)) -- conv1_1(64,3,1)
features:add(cudnn.ReLU(true)) -- relu1_1
features:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1)) -- conv1_2(64,3,1)
features:add(cudnn.ReLU(true)) -- relu1_2
features:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()) -- pool1(2,2)

features:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1, 1)) -- conv2_1(128,3,1)
features:add(cudnn.ReLU(true)) -- relu2_1
features:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1)) -- conv2_2(128,3,1)
features:add(cudnn.ReLU(true)) -- relu2_2
features:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()) -- pool2(2,2)

features:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 1)) -- conv3_1(256,3,1)
features:add(cudnn.ReLU(true)) -- relu3_1
features:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)) -- conv3_2(256,3,1)
features:add(cudnn.ReLU(true)) -- relu3_2
features:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)) -- conv3_3(256,3,1)
features:add(cudnn.ReLU(true)) -- relu3_3
features:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()) -- pool3(2,2)

features:add(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)) -- conv4_1(512,3,1)
features:add(cudnn.ReLU(true)) -- relu4_1
features:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)) -- conv4_2(512,3,1)
features:add(cudnn.ReLU(true)) -- relu4_2
features:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)) -- conv4_3(512,3,1)
features:add(cudnn.ReLU(true)) -- relu4_3
features:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()) -- pool4(2,2)

features:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)) -- conv5_1(512,3,1)
features:add(cudnn.ReLU(true)) -- relu5_1
features:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)) -- conv5_2(512,3,1)
features:add(cudnn.ReLU(true)) -- relu5_2
features:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)) -- conv5_3(512,3,1)
features:add(cudnn.ReLU(true)) -- relu5_3 -- Done features part

-- add ROIPooling after relu5_3
local ROIPooling = inn.ROIPooling(6,6):setSpatialScale(1,16) -- use default in paper/ in prototxt, it is 7?

-- classifier
classifier:add(nn.Linear(25088, 4096)) -- fc6
classifier:add(cudnn.ReLU(true))
classifier:add(nn.Dropout(0.5)) -- fc6 dropout
classifier:add(nn.Linear(4096, 4096)) -- fc7
classifier:add(cudnn.ReLU(true))
classifier:add(nn.Dropout(0.5)) -- fc7 dropout
classifier:add(nn.Linear(4096, num_classes)) -- classify

-- Input Table
local prl = nn.ParallelTable() 
prl:add(features)
prl:add(nn.Identity())

-- the final model
local model = nn.Sequential()
model:add(prl)
model:add(ROIPooling) -- ROIPooling
model:add(nn.View(-1):setNumInputDims(3))
model:add(classifier)


-- let's copy the pre-trained weights in caffemodel
-- pre-trained model: params
-- torch defined model: model
local lparams = model:parameters() -- get empty parameter tensors
for k,v in ipairs(lparams) do -- iterate over each layer
    local p = params[k] -- get pretrained layer weights
    assert(p:numel() == v:numel(), 'weights shape don\'t match')
    v:copy(p)
end

print ("Convertion Done!")

-- now let's save our model to torch format
torch.save('/home/lc/Caffe/caffemodels/fast-rcnn/frcnn_vgg16_flickr8k.t7', params) 

-- test the saved model 
local temp_model = torch.load('/home/lc/Caffe/caffemodels/fast-rcnn/frcnn_vgg16_flickr8k.t7')
print ("Converted model:", temp_model) io.read(1)


