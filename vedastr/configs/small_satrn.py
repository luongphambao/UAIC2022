###############################################################################
# 1. deploy

size = (32, 100)
mean, std = 0.5, 0.5

sensitive = True
character = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&''°()*+,-./:;<=>?@[\]^_`{|}~ '
test_sensitive = False
test_character ='aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&''°()*+,-./:;<=>?@[\]^_`{|}~ '
batch_max_length = 25

dropout = 0.1
n_e = 9
n_d = 3
hidden_dim = 256
n_head = 8
batch_norm = dict(type='BN')
layer_norm = dict(type='LayerNorm', normalized_shape=hidden_dim)
num_class = len(character) + 1
num_steps = batch_max_length + 1

deploy = dict(
    transform=[
        dict(type='Sensitive', sensitive=sensitive, need_character=character),
        dict(type='ToGray'),
        dict(type='Resize', size=size),
        dict(type='Normalize', mean=mean, std=std),
        dict(type='ToTensor'),
    ],
    converter=dict(
        type='AttnConverter',
        character=character,
        batch_max_length=batch_max_length,
        go_last=True,
    ),
    model=dict(
        type='GModel',
        need_text=True,
        body=dict(
            type='GBody',
            pipelines=[
                dict(
                    type='FeatureExtractorComponent',
                    from_layer='input',
                    to_layer='cnn_feat',
                    arch=dict(
                        encoder=dict(
                            backbone=dict(
                                type='GResNet',
                                layers=[
                                    ('conv',
                                     dict(type='ConvModule', in_channels=1, out_channels=int(hidden_dim / 2),
                                          kernel_size=3,
                                          stride=1, padding=1, norm_cfg=batch_norm)),
                                    ('pool', dict(type='MaxPool2d', kernel_size=2, stride=2, padding=0)),
                                    ('conv',
                                     dict(type='ConvModule', in_channels=int(hidden_dim / 2), out_channels=hidden_dim,
                                          kernel_size=3,
                                          stride=1, padding=1, norm_cfg=batch_norm)),
                                    ('pool', dict(type='MaxPool2d', kernel_size=2, stride=2, padding=0)),
                                ],
                            ),
                        ),
                        collect=dict(type='CollectBlock', from_layer='c2'),
                    ),
                ),
                dict(
                    type='SequenceEncoderComponent',
                    from_layer='cnn_feat',
                    to_layer='src',
                    arch=dict(
                        type='TransformerEncoder',
                        position_encoder=dict(
                            type='Adaptive2DPositionEncoder',
                            in_channels=hidden_dim,
                            max_h=100,
                            max_w=100,
                            dropout=dropout,
                        ),
                        encoder_layer=dict(
                            type='TransformerEncoderLayer2D',
                            attention=dict(
                                type='MultiHeadAttention',
                                in_channels=hidden_dim,
                                k_channels=hidden_dim // n_head,
                                v_channels=hidden_dim // n_head,
                                n_head=n_head,
                                dropout=dropout,
                            ),
                            attention_norm=layer_norm,
                            feedforward=dict(
                                type='Feedforward',
                                layers=[
                                    dict(type='ConvModule', in_channels=hidden_dim, out_channels=hidden_dim * 4,
                                         kernel_size=3, padding=1,
                                         bias=True, norm_cfg=None, activation='relu', dropout=dropout),
                                    dict(type='ConvModule', in_channels=hidden_dim * 4, out_channels=hidden_dim,
                                         kernel_size=3, padding=1,
                                         bias=True, norm_cfg=None, activation=None, dropout=dropout),
                                ],
                            ),
                            feedforward_norm=layer_norm,
                        ),
                        num_layers=n_e,
                    ),
                ),
            ],
        ),
        head=dict(
            type='TransformerHead',
            src_from='src',
            num_steps=num_steps,
            pad_id=num_class,
            decoder=dict(
                type='TransformerDecoder',
                position_encoder=dict(
                    type='PositionEncoder1D',
                    in_channels=hidden_dim,
                    max_len=100,
                    dropout=dropout,
                ),
                decoder_layer=dict(
                    type='TransformerDecoderLayer1D',
                    self_attention=dict(
                        type='MultiHeadAttention',
                        in_channels=hidden_dim,
                        k_channels=hidden_dim // n_head,
                        v_channels=hidden_dim // n_head,
                        n_head=n_head,
                        dropout=dropout,
                    ),
                    self_attention_norm=layer_norm,
                    attention=dict(
                        type='MultiHeadAttention',
                        in_channels=hidden_dim,
                        k_channels=hidden_dim // n_head,
                        v_channels=hidden_dim // n_head,
                        n_head=n_head,
                        dropout=dropout,
                    ),
                    attention_norm=layer_norm,
                    feedforward=dict(
                        type='Feedforward',
                        layers=[
                            dict(type='FCModule', in_channels=hidden_dim, out_channels=hidden_dim * 4, bias=True,
                                 activation='relu', dropout=dropout),
                            dict(type='FCModule', in_channels=hidden_dim * 4, out_channels=hidden_dim, bias=True,
                                 activation=None, dropout=dropout),
                        ],
                    ),
                    feedforward_norm=layer_norm,
                ),
                num_layers=n_d,
            ),
            generator=dict(
                type='Linear',
                in_features=hidden_dim,
                out_features=num_class,
            ),
            embedding=dict(
                type='Embedding',
                num_embeddings=num_class + 1,
                embedding_dim=hidden_dim,
                padding_idx=num_class,
            ),
        ),
    ),
    postprocess=dict(
        sensitive=test_sensitive,
        character=test_character,
    ),
)

###############################################################################
# 2.common

common = dict(
    seed=1111,
    logger=dict(
        handlers=(
            dict(type='StreamHandler', level='INFO'),
            dict(type='FileHandler', level='INFO'),
        ),
    ),
    cudnn_deterministic=True,
    cudnn_benchmark=True,
    metric=dict(type='Accuracy'),
)

###############################################################################
dataset_params = dict(
    batch_max_length=batch_max_length,
    data_filter=True,
    character=character,
)
test_dataset_params = dict(
    batch_max_length=batch_max_length,
    data_filter=False,
    character=test_character,
)



###############################################################################
# 3. test

batch_size = 128

# data

###############################################################################
# 4. train

root_workdir = 'weights'  # save directory

# data
data_root = 'data/'
train_root = data_root + "train_rec_data"
# MJ dataset
train_root_mj = train_root + ''
mj_folder_names = []
# ST dataset
train_root_st = train_root + ''

train_dataset_mj = [dict(type='LmdbDataset', root=train_root_mj)]
train_dataset_st = [dict(type='LmdbDataset', root=train_root_st)]


# valid
valid_root = data_root + "valid_rec_data"
valid_dataset = dict(type='LmdbDataset', root=valid_root, **dataset_params)

train_transforms = [
    dict(type='Sensitive', sensitive=sensitive, need_character=character),
    dict(type='ToGray'),
    dict(type='ExpandRotate', limit=34, p=0.5),
    dict(type='Resize', size=size),
    dict(type='Normalize', mean=mean, std=std),
    dict(type='ToTensor'),
]

max_epochs = 100
milestones = [2, 4]  # epoch start from 0, so 2 means lr decay at 3 epoch, 4 means lr decay at the end of

train = dict(
    data=dict(
        train=dict(
            dataloader=dict(
                type='DataLoader',
                batch_size=batch_size,
                num_workers=4,
            ),
            sampler=dict(
                type='BalanceSampler',
                batch_size=batch_size,
                shuffle=True,
                oversample=True,
            ),
            dataset=dict(
                type='ConcatDatasets',
                datasets=[
                    dict(
                        type='ConcatDatasets',
                        datasets=train_dataset_mj,
                    ),
                    dict(
                        type='ConcatDatasets',
                        datasets=train_dataset_st,
                    )
                ],
                batch_ratio=[0.5, 0.5],
                **dataset_params,
            ),
            transform=train_transforms,
        ),
        val=dict(
            dataloader=dict(
                type='DataLoader',
                batch_size=batch_size,
                num_workers=4,
                shuffle=False,
            ),
            dataset=valid_dataset,
            transform=deploy['transform'],
        ),
    ),
    optimizer=dict(type='Adam', lr=3e-4),
    criterion=dict(type='CrossEntropyLoss', ignore_index=num_class),
    lr_scheduler=dict(type='CosineLR',
                      iter_based=True,
                      warmup_epochs=0.1,
                      ),
    max_epochs=max_epochs,
    log_interval=10,
    trainval_ratio=2000,
    snapshot_interval=20000,
    save_best=True,
    resume=None,
)
