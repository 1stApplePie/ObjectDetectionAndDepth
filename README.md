# 자율주행 데브코스 2차 오프라인 프로젝트 : Team 6 삼대장

## <프로젝트 목표>

+ 목표 1. 객체를 인식하여 트랙을 자율주행
  
  + 모든 차량을 출발지점의 정지선에서 신호 대기
  + 신호등의 출발 신호에 맞춰 출발
  + 제시되는 표지판의 내용대로 정지, 좌, 우 조향 제어
  + 동적 장애물은 갑작스레 등장할 예정
    
+ 목표 2. 차량주변의 객체 위치정보를 Bird's Eye View로 표현
  
  + 트랙을 주행하며 주변 객체의 정보를 인식
  + BEV에 객체 정보 표현
  + 객체는 2D로 X, Y 좌표만 표현(Point로 표현)

## Dependency

pip install -r requirements.txt
```
python >= 3.6

Numpy

torch >= 1.9

torchvision >= 0.10

tensorboard

tensorboardX

torchsummary

pynvml

imgaug

onnx

onnxruntime
```

-------------------

## Run
"-- checkout" arg needs pre-trained model, so if you run first time, don't need to set checkpoint arg<br>
If using gpu set "-- gpus" 0, 1, 2, 3<br>

If training,

```{r, engine='bash', count_lines}
(no gpu) python main.py --mode train --cfg ./yolov3_kitti.cfg

(single gpu) python main.py --mode train --cfg ./yolov3_kitti.cfg --gpus 0

(multi gpu) python main.py --mode train --cfg ./yolov3_kitti.cfg --gpus 0 1 2 3

(transfer learning) use --checkpoint ${saved_checkpoint_path}
```

If evaluate,

```{r, engine='bash', count_lines}
python main.py --mode eval --cfg ./yolov3_kitti.cfg --checkpoint ${saved_checkpoint_path}
```

If inference,

```{r, engine='bash', count_lines}
python main.py --mode demo --cfg ./yolov3_kitti.cfg --checkpoint ${saved_checkpoint_path}
```

If converting torch to onnx,

target tensorrt version > 7
```{r, engine='bash', count_lines}
python main.py --mode onnx --cfg ./cfg/yolov3.cfg --gpus 0 --checkpoint ${saved_checkpoint_path}
```

target tensorrt version is 5.x

1. **ONNX_EXPORT = True** in 'model/yolov3.py'
   
   tensorrt(v5.x) is not support upsample scale factor, so you have to change upsample layer not using scale factor.

```{r, engine='bash', count_lines}
python main.py --mode onnx --cfg ./cfg/yolov3.cfg --gpus 0 --checkpoint ${saved_checkpoint_path}
```

### option

--mode : train/eval/demo/auto-label.

--cfg : the path of model.cfg.

--gpu : if you use GPU, set 1. If you use CPU, set 0.

--checkpoint (optional) : the path of saved model checkpoint. Use it when you want to load the previous train, or you want to test(evaluate) the model.

--pretrained (optional) : the path of darknet pretrained weights. Use it when you want to fine-tuning the model.



## Visualize training graph

Using Tensorboard,

```{r, engine='bash', count_lines}
tensorboard --logdir=./output --port 8888
```

-------------------------

# Reference

[YOLOv3 paper](https://arxiv.org/abs/1804.02767)

[KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)


# error

- if libgl.so error when cv2
```
apt-get update
apt-get install libgl1-mesa-glx
```
