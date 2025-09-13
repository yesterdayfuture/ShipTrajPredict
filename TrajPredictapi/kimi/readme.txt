
pipeline整体流程：
    全局归一化（不再按轨迹单独归一化）；
    输入特征 = [lat, lon, sog, cog_sin, cog_cos, type_emb]；
    输出 = 未来 HORIZON 步的 Δlat, Δlon；
    Transformer 输出取最后一个时间步后接 2×HORIZON 维回归头；
    训练用 HuberLoss，评估用 ADE/FDE；
    支持批量反归一化（不再把 min/max 当特征，而是推理时直接加回）。

当前文件夹模型说明：
    kimi_model.pth
        网络结构：
            TrajTransformer
        参数：
            SEQ_LEN  = 32
            HORIZON  = 6
            BATCH    = 256
            EPOCHS   = 3
        评估结果：
            ADE(norm)=0.0007, FDE(norm)=0.0013
            ADE=2.390 nmi, FDE=4.428 nmi
            x size torch.Size([1, 32, 5]) delta size torch.Size([6, 2])
            真实经纬度：tensor([ 25.8037, -77.4094])  预测经纬度： tensor([ 25.8099, -77.4331])
            真实经纬度：tensor([ 25.8057, -77.4134])  预测经纬度： tensor([ 25.8191, -77.3739])
            真实经纬度：tensor([ 25.8075, -77.4169])  预测经纬度： tensor([ 25.7952, -77.4176])
            真实经纬度：tensor([ 25.8096, -77.4211])  预测经纬度： tensor([ 25.7972, -77.3582])
            真实经纬度：tensor([ 25.8116, -77.4251])  预测经纬度： tensor([ 25.8071, -77.3859])
            真实经纬度：tensor([ 25.8136, -77.4292])  预测经纬度： tensor([ 25.7861, -77.3440])

    kimi_model2.pth
        网络结构：
            TrajTransformer
        参数：
            SEQ_LEN  = 32
            HORIZON  = 6
            BATCH    = 256
            EPOCHS   = 30
        评估结果：
            ADE(norm)=0.0002, FDE(norm)=0.0004
            ADE=0.757 nmi, FDE=1.176 nmi
            x size torch.Size([1, 32, 5]) delta size torch.Size([6, 2])
            真实经纬度：tensor([ 25.8037, -77.4094])  预测经纬度： tensor([ 25.8027, -77.3969])
            真实经纬度：tensor([ 25.8057, -77.4134])  预测经纬度： tensor([ 25.8024, -77.4042])
            真实经纬度：tensor([ 25.8075, -77.4169])  预测经纬度： tensor([ 25.8063, -77.3990])
            真实经纬度：tensor([ 25.8096, -77.4211])  预测经纬度： tensor([ 25.8081, -77.4067])
            真实经纬度：tensor([ 25.8116, -77.4251])  预测经纬度： tensor([ 25.8142, -77.4058])
            真实经纬度：tensor([ 25.8136, -77.4292])  预测经纬度： tensor([ 25.8140, -77.4053])

    kimi_model3.pth
        网络结构：
            TrajTransformer2
        参数：
            SEQ_LEN  = 64
            HORIZON  = 6
            BATCH    = 256
            EPOCHS   = 10
        评估结果：
            ADE(norm)=0.0004, FDE(norm)=0.0006
            ADE=1.047 nmi, FDE=1.620 nmi
            x size torch.Size([1, 64, 5]) delta size torch.Size([6, 2])
            真实经纬度：tensor([ 25.8795, -77.5307])  预测经纬度： tensor([ 25.8701, -77.5224])
            真实经纬度：tensor([ 25.8822, -77.5343])  预测经纬度： tensor([ 25.8689, -77.5234])
            真实经纬度：tensor([ 25.8848, -77.5377])  预测经纬度： tensor([ 25.8679, -77.5223])
            真实经纬度：tensor([ 25.8875, -77.5413])  预测经纬度： tensor([ 25.8679, -77.5238])
            真实经纬度：tensor([ 25.8901, -77.5448])  预测经纬度： tensor([ 25.8667, -77.5241])
            真实经纬度：tensor([ 25.8928, -77.5484])  预测经纬度： tensor([ 25.8681, -77.5249])

    kimi_model4.pth
            网络结构：
                TrajTransformer2
            参数：
                SEQ_LEN  = 32
                HORIZON  = 6
                BATCH    = 128
                EPOCHS   = 13
            评估结果：
                ADE(norm)=0.0004, FDE(norm)=0.0006
                ADE=0.940 nmi, FDE=1.609 nmi
                x size torch.Size([1, 32, 5]) delta size torch.Size([6, 2])
                真实经纬度：tensor([ 25.8037, -77.4094])  预测经纬度： tensor([ 25.8022, -77.4047])
                真实经纬度：tensor([ 25.8057, -77.4134])  预测经纬度： tensor([ 25.8034, -77.4047])
                真实经纬度：tensor([ 25.8075, -77.4169])  预测经纬度： tensor([ 25.8037, -77.4037])
                真实经纬度：tensor([ 25.8096, -77.4211])  预测经纬度： tensor([ 25.8045, -77.4044])
                真实经纬度：tensor([ 25.8116, -77.4251])  预测经纬度： tensor([ 25.8048, -77.4034])
                真实经纬度：tensor([ 25.8136, -77.4292])  预测经纬度： tensor([ 25.8059, -77.4046])

    kimi_model5.pth
            网络结构：
                TrajTransformer
            参数：
                SEQ_LEN  = 32
                HORIZON  = 6
                BATCH    = 128
                EPOCHS   = 30
            评估结果：
                ADE(norm)=0.0002, FDE(norm)=0.0004
                ADE=0.701 nmi, FDE=1.208 nmi
                x size torch.Size([1, 32, 5]) delta size torch.Size([6, 2])
                真实经纬度：tensor([ 25.8037, -77.4094])  预测经纬度： tensor([ 25.8034, -77.4053])
                真实经纬度：tensor([ 25.8057, -77.4134])  预测经纬度： tensor([ 25.8030, -77.4064])
                真实经纬度：tensor([ 25.8075, -77.4169])  预测经纬度： tensor([ 25.8052, -77.4041])
                真实经纬度：tensor([ 25.8096, -77.4211])  预测经纬度： tensor([ 25.8051, -77.4022])
                真实经纬度：tensor([ 25.8116, -77.4251])  预测经纬度： tensor([ 25.8073, -77.4065])
                真实经纬度：tensor([ 25.8136, -77.4292])  预测经纬度： tensor([ 25.8062, -77.4069])


    kimi_model6.pth
            网络结构：
                TrajTransformer3
            参数：
                SEQ_LEN  = 32
                HORIZON  = 6
                BATCH    = 128
                EPOCHS   = 60
            评估结果：
                ADE(norm)=0.0002, FDE(norm)=0.0003
                ADE=0.668 nmi, FDE=1.160 nmi
                x size torch.Size([1, 32, 5]) delta size torch.Size([6, 2])
                真实经纬度：tensor([ 25.8037, -77.4094])  预测经纬度： tensor([ 25.8032, -77.4055])
                真实经纬度：tensor([ 25.8057, -77.4134])  预测经纬度： tensor([ 25.8054, -77.4061])
                真实经纬度：tensor([ 25.8075, -77.4169])  预测经纬度： tensor([ 25.8065, -77.4053])
                真实经纬度：tensor([ 25.8096, -77.4211])  预测经纬度： tensor([ 25.8080, -77.4058])
                真实经纬度：tensor([ 25.8116, -77.4251])  预测经纬度： tensor([ 25.8093, -77.4035])
                真实经纬度：tensor([ 25.8136, -77.4292])  预测经纬度： tensor([ 25.8115, -77.4025])







