# DirSpread-VLA

训练无关。视频上面是原版，下面先看碗，夹紧并移动后看盘子，速度变化后再看碗。摊开的是图像 token 的方向，长度保持原样。

模型 `gs://openpi-assets/checkpoints/pi05_libero`  
分词器 `gs://big_vision/paligemma_tokenizer.model`  

第一次运行会自动下载这两个文件。需要本机的 openpi、LIBERO、ffmpeg、gsutil。

```bash
OPENPI_ROOT=/path/to/openpi ./run.sh
```

`tasks.json` 是失败合集的 23 段。这条命令按段号逐段生成，视频写到 `videos/compare_seg01.mp4` 到 `videos/compare_seg23.mp4`。已经生成的文件会跳过。

只重做其中几段：

```bash
OPENPI_ROOT=/path/to/openpi ./run.sh 5 11 12
```
