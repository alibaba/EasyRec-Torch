# 特征

TorchEasyRec多种类型的特征，包括IdFeature、RawFeature、ComboFeature、LookupFeature、MatchFeature、ExprFeature、OverlapFeature、TokenizeFeature、SequenceIdFeature、SequenceRawFeature、SequenceFeature。

**共用配置**

- **feature_name**: 特征名/特征输出名

- **embedding_dim**: 特征嵌入维度，计算方法可以参考: `embedding_dim=8+x^{0.25}`，`embedding_dim` 需要为4的倍数, 其中，x 为不同特征取值的个数

- **embedding_name**: 特征嵌入名，默认不需要设置，如需两个特征共享嵌入(embedding)参数，可将这两个特征的嵌入名设置成相同

- **pooling**: 多值特征嵌入池化方式，默认为`sum`，可以选`sum`/`mean`

- **init_fn**: 特征嵌入初始化方式，默认不需要设置，如需自定义，可以设置任意的torch内置初始化函数，如`nn.init.uniform_,a=-0.01,b=0.01`

- **default_value**: 特征默认值。如果默认值为""，则没有默认值，后续模型中对于空特征的嵌入为零向量。注意: 该默认值为`bucktize`前的默认值。`bucktize`的配置包括`hash_bucket_size`/`vocab_list`/`boundaries`

- **separator**: FG在输入为string类型时的多值分隔符，默认为`\x1d`。更建议用数组（ARRAY）类型来表示多值，训练和推理性能更好

- **fg_encoded_default_value**: FG编码后的数据的默认值，当fg_encoded=true并且不是用pai-fg编码数据时，可以设置该参数填充空值

## IdFeature: 类别型特征

类别型特征，例如手机品牌、item_id、user_id、年龄段、星座等，一般在表里面存储的类型一般是string、bigint、array<string>或array<bigint>。可支持多值Id特征

```
feature_configs {
    id_feature {
        feature_name: "uid"
        # fg_encoded=true的情况下，expression可以不写
        expression: "user:uid"
        embedding_dim: 32
        hash_bucket_size: 100000
    }
}
feature_configs {
    id_feature {
        feature_name: "month"
        expression: "user:month"
        embedding_dim: 32
        num_buckets: 12
        default_value: "0"
    }
}
feature_configs {
    id_feature {
        feature_name: "cate"
        expression: "item:cate"
        embedding_dim: 32
        vocab_list: ["1", "2", "3", "4", "5", "6", "7"]
    }
}
feature_configs {
    id_feature {
        feature_name: "cate"
        expression: "item:cate"
        embedding_dim: 32
        vocab_dict: [{key:"a" value:2}, {key:"b" value:3}, {key:"c" value:2}]
    }
feature_configs {
    id_feature {
        feature_name: "cate"
        expression: "item:cate"
        embedding_dim: 32
        zch: {
            zch_size: 1000000
            eviction_interval: 2
            lfu {}
        }
    }
}
```

- **expression**: 特征FG所依赖的字段来源，由两部分组成`input_side`:`input_name`

  - `input_side`可以取值为`user`/`item`/`context`，用于指导推荐模型服务从哪里获取相关特征，`user`表示这个特征是用户侧特征，从请求中获取，`item`表示是物品侧特征，可以从模型服务内存Cache中获取，`context`表示是需从请求里传递的物品侧特征，如召回名等
  - `input_name`为来源字段的实际名称

- **hash_bucket_size**: hash bucket的大小。为减少hash冲突，建议设置
  `hash_bucket_size  = number_ids*ratio, ratio in [5,10]`

- **num_buckets**: buckets数量, 仅仅当输入是integer类型时，可以使用num_buckets

- **vocab_list**: 指定词表，适合取值比较少可以枚举的特征，如星期，月份，星座等，**编号需要从2开始**，编码0预留给默认值，编码1预留给超出词表的词

- **vocab_dict**: 指定字典形式词表，适合多个词需要编码到同一个编号情况，**编号需要从2开始**，编码0预留给默认值，编码1预留给超出词表的词

- **zch**: 零冲突hash，可设置Id的准入和驱逐策略，详见[文档](../zch.md)

- **weighted**: 是否为带权重的Id特征，输入形式为`k1:v1\x1dk2:v2`

- **value_dim**: 默认值是0，可以设置1，value_dim=0时支持多值ID输出

- **default_bucketize_value**: （可选）指定超出词表的词的编码。当配置了default_bucketize_value时，vocab_list和vocab_dict将不会预留编码给默认值和超出词表的词，用户可完全自主控制vocab_list或vocab_dict

- NOTE: hash_bucket_size, num_buckets, vocab_list, 只能指定其中之一，不能同时指定

## RawFeature: 数值型特征

```
feature_configs {
    raw_feature {
        feature_name: "ctr"
        expression: "item:ctr"
    }
}
```

- **expression**: 特征FG所依赖的字段来源，由两部分组成`input_side`:`input_name`，`input_side`可以取值为`user`/`item`/`context`，`input_name`为来源字段的名称

- **normalizer**: 指定连续值特征的变换方式，支持4种，默认不变换

  - log10

  ```
  配置例子: method=log10,threshold=1e-10,default=-10
  计算公式: x = x > threshold ? log10(x) : default;
  ```

  - zscore

  ```
  配置例子: method=zscore,mean=0.0,standard_deviation=10.0
  计算公式: x = (x - mean) / standard_deviation
  ```

  - minmax

  ```
  配置例子: method=minmax,min=2.1,max=2.2
  计算公式: x = (x - min) / (max - min)
  ```

  - expression，表达式配置方式详见**ExprFeature**

  ```
  配置例子: method=expression,expr=sign(x)
  计算公式: 可以配置任意的函数或表达式，变量名固定为x，代表表达式的输入
  ```

连续值类特征可以先使用分箱组件+进行离散化，可以进行等频/等距/自动离散化，变成离散值。分箱组件使用方法见:
[机器学习组件](https://help.aliyun.com/zh/pai/user-guide/binning)

```
feature_configs {
    raw_feature {
        feature_name: "ctr"
        expression: "item:ctr"
        boundaries: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        embedding_dim: 8
    }
}
feature_configs {
    raw_feature {
        feature_name: "price"
        expression: "item:price"
        embedding_dim: 8
        mlp {}
    }
}
feature_configs {
    raw_feature {
        feature_name: "price"
        expression: "item:price"
        embedding_dim: 8
        autodis {
           num_channels: 3
           temperature: 0.1
           keep_prob: 0.8
        }
    }
}
```

- **boundaries**: 分箱/分桶的边界值，通过一个数组来设置。
- **mlp**: 由一层MLP变换特征到`embedding_dim`维度
- **autodis**: 由AutoDis模块变换特征到`embedding_dim`维度，详见[AutoDis文档](../autodis.md)

Embedding特征: 支持string类型如`"0.1|0.2|0.3|0.4"`；支持ARRAY<float>类型如`[0.1,0.2,0.3,0.4]`（建议，性能更好），配置方式如下

```
feature_configs: {
    raw_feature {
        feature_name: "pic_emb"
        expression: "item:pic_emb"
        feature_type: RawFeature
        separator: "|"
        value_dim: 4
    }
}
```

- **separator**: FG多值分隔符，默认为`\x1d`
- **value_dim**: 指定Embedding特征的输入维度

## ComboFeature: 组合特征

对输入的离散值进行组合（即笛卡尔积）, 如age + cate:

```
feature_configs: {
    combo_feature {
        feature_name: "combo_age_cate"
        expression: ["user:age", "item:cate"]
        embedding_dim: 16
        hash_bucket_size: 1000
    }
}
```

- expression: 特征FG所依赖组合字段的来源，数量 >= 2
- 其余配置同IdFeature，NOTE: ComboFeature不包含`num_buckets`配置

## LookupFeature: 字典查询特征

`lookup_feature`依赖`map`和`key`两个字段从一组kv中匹配特征值。生成特征时，使用key在map字段所持有的kv对中进行匹配，获取最终的特征。

`map`是一个多值的kv map，支持string类型，如`"k1:v1\x1dk2:v2"`，其中`:`为kv分割符，`\x1d`为固定的多值分隔符；也支持MAP\<string,bigint>/MAP\<bigint,bigint>等MAP类型，如`{"k1":v1,"k2":v2}`（建议，性能更好）

`key`是一个多值的id，多值分隔符可以由**separator**指定，默认为`\x1d`。

```
feature_configs: {
    lookup_feature {
        feature_name: "user_cate_cnt"
        map: "user:kv_cate_cnt"
        key: "item:cate"
        embedding_dim: 16
        boundaries: [0, 1, 2, 3, 4]
    }
}
```

- **map**: 特征FG所依赖map字段的来源
- **key**: 特征FG所依赖key字段的来源
- **combiner**: 如果key为多值，可以设置combiner来对查找的值进行聚合，默认为`sum`，支持`sum`/`mean`/`min`/`max`
- **need_discrete**: 查到的值是否为离散值，默认为false
- **need_key**: 查到的值是否拼接key作为前缀，默认为false

如果Map的值为离散值 或 `need_key=true`，可设置:

- **hash_bucket_size**: hash bucket的大小。
- **num_buckets**: buckets数量, 仅仅当输入是integer类型时，可以使用num_buckets
- **vocab_list**: 指定词表，适合取值比较少可以枚举的特征。
- **vocab_dict**: 指定字典形式词表，适合多个词需要编码到同一个编号情况，**编号需要从2开始**，编码0预留给默认值，编码1预留给超出词表的词
- **zch**: 零冲突hash，可设置Id的准入和驱逐策略，详见[文档](../zch.md)
- **value_dim**: 默认值是0，可以设置1，value_dim=0时支持多值ID输出

如果Map的值为连续值，可设置:

- **boundaries**: 分箱/分桶的值。
- **normalizer**: 连续值特征的变换方式，同RawFeature
- **value_dim**: 默认值是1，连续值输出维度
- **value_separator**: 连续值分隔符
- **mlp**: 由一层MLP变换特征到`embedding_dim`维度
- **autodis**: 由AutoDis模块变换特征到`embedding_dim`维度，详见[AutoDis文档](../autodis.md)

## MatchFeature: 主从键字典查询特征

`match_feature`依赖`nested_map`和`pkey`和`skey`三个字段从kkv中匹配到特征值。`nested_map`是一个多值的kkv map，如`pk1^sk1:0.2,sk2:0.3,sk3:0.5|pk2^sk4:0.1`，`:`为内层kv分割符，`,`为内层多值分隔符，`^`为外层kv分割符，`|`为外层KV分隔符，分隔符不可以指定。生成特征时，使用`pkey`作为主键`skey`作为子健在`nested_map`字段所持有的kkv对中进行匹配，获取最终的特征。

```
feature_configs: {
    match_feature {
        feature_name: "user_cate_brand_cnt"
        nested_map: "user:kkv_cate_brand_cnt"
        pkey: "item:cate"
        skey: "item:brand"
        embedding_dim: 16
        boundaries: [0, 1, 2, 3, 4]
    }
}
```

- **nested_map**: 特征FG所依赖nested_map字段的来源
- **pkey**: 特征FG所依赖主键字段的来源。可以设置为ALL，将匹配所有pkey下面的指定的skey的值
- **skey**: 特征FG所依赖子键字段的来源。可以设置为ALL，将指定pkey下面所有skey的值
- **combiner**: 如果key为多值，可以设置combiner来对查找的值进行聚合，支持`sum`/`mean`/`min`/`max`
- **need_discrete**: 查到的值是否为离散值，默认为false
- **show_pkey**: 查到的值是否拼接pkey作为前缀，默认为false
- **show_skey**: 查到的值是否拼接skey作为前缀，默认为false

如果Map的值为离散值 或 `show_pkey=true` 或 `show_skey=true`，可设置:

- **hash_bucket_size**: hash bucket的大小。
- **num_buckets**: buckets数量, 仅仅当输入是integer类型时，可以使用num_buckets
- **vocab_list**: 指定词表，适合取值比较少可以枚举的特征。
- **vocab_dict**: 指定字典形式词表，适合多个词需要编码到同一个编号情况，**编号需要从2开始**，编码0预留给默认值，编码1预留给超出词表的词
- **zch**: 零冲突hash，可设置Id的准入和驱逐策略，详见[文档](../zch.md)
- **value_dim**: 默认值是0，可以设置1，value_dim=0时支持多值ID输出

如果Map的值为连续值，可设置:

- **boundaries**: 分箱/分桶的值。
- **normalizer**: 连续值特征的变换方式，同RawFeature
- **value_dim**: 目前只支持value_dim=1
- **mlp**: 由一层MLP变换特征到`embedding_dim`维度
- **autodis**: 由AutoDis模块变换特征到`embedding_dim`维度，详见[AutoDis文档](../autodis.md)

## ExprFeature: 表达式特征

对数值型特征进行运算，如判断当前用户年龄是否>18，用户年龄是否符合物品年龄需求等。

```
feature_configs: {
    expr_feature {
        feature_name: "combo_age_cate"
        variables: ["user:u_age", "item:i_age"]
        expression: "u_age == i_age ? 1 : 0"
        embedding_dim: 8
        boundaries: [0.5]
    }
}
```

- **variables**: 特征FG所依赖表达式中的字段的来源

- **expression**: 表达式本身

- **内置函数**:

  | 函数名      | 参数数量 | 解释                                   |
  | ----------- | -------- | -------------------------------------- |
  | sin         | 1        | sine function                          |
  | cos         | 1        | cosine function                        |
  | tan         | 1        | tangens function                       |
  | asin        | 1        | arcus sine function                    |
  | acos        | 1        | arcus cosine function                  |
  | atan        | 1        | arcus tangens function                 |
  | sinh        | 1        | hyperbolic sine function               |
  | cosh        | 1        | hyperbolic cosine                      |
  | tanh        | 1        | hyperbolic tangens function            |
  | asinh       | 1        | hyperbolic arcus sine function         |
  | acosh       | 1        | hyperbolic arcus tangens function      |
  | atanh       | 1        | hyperbolic arcur tangens function      |
  | log2        | 1        | logarithm to the base 2                |
  | log10       | 1        | logarithm to the base 10               |
  | log         | 1        | logarithm to base e (2.71828...)       |
  | ln          | 1        | logarithm to base e (2.71828...)       |
  | exp         | 1        | e raised to the power of x             |
  | sqrt        | 1        | square root of a value                 |
  | sign        | 1        | sign function -1 if x\<0; 1 if x>0     |
  | rint        | 1        | round to nearest integer               |
  | abs         | 1        | absolute value                         |
  | sigmoid     | 1        | sigmoid function                       |
  | l2_norm     | 1        | l2 normalize of a vector               |
  | dot         | 2        | dot product of two vectors             |
  | euclid_dist | 2        | euclidean distance between two vectors |
  | min         | var.     | min of all arguments                   |
  | max         | var.     | max of all arguments                   |
  | sum         | var.     | sum of all arguments                   |
  | avg         | var.     | mean value of all arguments            |

- **内置二元操作符**:

  | 操作符 | 描述                      | 优先级 |
  | ------ | ------------------------- | ------ |
  | \|\|   | logical or                | 1      |
  | &&     | logical and               | 2      |
  | \|     | bitwise or                | 3      |
  | &      | bitwise and               | 4      |
  | \<=    | less or equal             | 5      |
  | >=     | greater or equal          | 5      |
  | !=     | not equal                 | 5      |
  | ==     | equal                     | 5      |
  | >      | greater than              | 5      |
  | \<     | less than                 | 5      |
  | +      | addition                  | 6      |
  | -      | subtraction               | 6      |
  | \*     | multiplication            | 7      |
  | /      | division                  | 7      |
  | ^      | raise x to the power of y | 8      |

- **内置三元操作符**

  | 操作符 | 描述                  | 优先级           |
  | ------ | --------------------- | ---------------- |
  | ?:     | if then else operator | C++ style syntax |

- **内置常量**

  | 操作符 | 描述                 | 优先级                     |
  | ------ | -------------------- | -------------------------- |
  | \_pi   | The one and only pi. | 3.141592653589793238462643 |
  | \_e    | Euler's number.      | 2.718281828459045235360287 |

- 其余配置同RawFeature

## OverlapFeature: 重合匹配特征

`overlap_feature`会计算`query`和`title`两个字段字词重合比例，`query`和`title`中字词的分割符默认为`\x1d`，可以用多值分隔符由**separator**指定。

```
feature_configs: {
    overlap_feature {
        feature_name: "user_cate_cnt"
        query: "user:query"
        title: "item:title"
        method: "title_common_ratio"
        embedding_dim: 8
        boundaries: [0.2, 0.5, 0.8]
    }
}
```

- **query**: 特征FG所依赖query字段的来源

- **title**: 特征FG所依赖title字段的来源

- **method**: 重合计算方式，可选 query_common_ratio | title_common_ratio | is_contain | is_equal

  | 方式               | 描述                                          | 备注                           |
  | ------------------ | --------------------------------------------- | ------------------------------ |
  | query_common_ratio | 计算query与title间重复term数占query中term比例 | 取值为\[0,1\]                  |
  | title_common_ratio | 计算query与title间重复term数占title中term比例 | 取值为\[0,1\]                  |
  | is_contain         | 计算query是否全部包含在title中，保持顺序      | 0表示未包含，1表示包含         |
  | is_equal           | 计算query是否与title完全相同                  | 0表示不完全相同，1表示完全相同 |

- 其余配置同RawFeature

## TokenizeFeature: 分词特征

`tokenize_feature` 对输入字符串分词，返回分词之后的词id。支持tokenize-cpp的分词词典文件。

```
feature_configs: {
    tokenize_feature {
        feature_name: "title_token"
        expression: "item:title"
        vocab_file: "tokenizer.json"
        embedding_dim: 8
        text_normalizer: {
            norm_options: [TEXT_LOWER2UPPER, TEXT_SBC2DBC, TEXT_CHT2CHS, TEXT_FILTER]
        }
    }
}
```

- **expression**: 特征FG所依赖分词字段的来源

- **vocab_file**: 分词字典，完全兼容 https://github.com/mlc-ai/tokenizers-cpp 库的分词文件

- **tokenizer_type**: 分词类型，支持bpe、sentencepiece，默认为bpe

- **text_normalizer**: 可选，是否对文本进行归一化

  - **stop_char_file**: 停用词表路径，默认为系统内置，详见[stop_char](https://tzrec.oss-cn-beijing.aliyuncs.com/third_party/stop_char)
  - **norm_options**: 归一化选项，默认为TEXT_LOWER2UPPER, TEXT_SBC2DBC, TEXT_CHT2CHS, TEXT_FILTER

  | 方式              | 描述                   |
  | ----------------- | ---------------------- |
  | TEXT_LOWER2UPPER  | 小写转换成大写         |
  | TEXT_UPPER2LOWER  | 大写转换成小写         |
  | TEXT_SBC2DBC      | 全角到半角             |
  | TEXT_CHT2CHS      | 繁体到简体             |
  | TEXT_FILTER       | 去除特殊符号           |
  | TEXT_SPLITCHRS    | 中文拆成单字(空格分隔) |
  | TEXT_REMOVE_SPACE | 去除空格               |

## SequenceIdFeature：类别型序列特征

类别型序列特征，支持string类型`item_id1;item_id2;item_id3`， 其中`;`为序列分隔符；支持ARRAY<string>或ARRAY<bigint>类型为`[item_id1,item_id2,item_id3]`（建议，性能更好）

```
feature_configs: {
    sequence_id_feature {
        feature_name: "click_itemid_seq"
        sequence_length: 50
        sequence_delim: ";"
        expression: "user:click_iid_seq"
        embedding_dim: 32
        hash_bucket_size: 100000
    }
}
```

- **sequence_length**: 序列特征最大长度
- **sequence_delim**: 序列特征分隔符
- **expression**: 特征FG所依赖的字段来源，由两部分组成`input_side`:`input_name`
- **value_dim**: 目前只支持value_dim=1，不支持多值ID序列
- 其余配置同IdFeature

## SequenceRawFeature：数值型序列特征

数值型序列特征，支持string类型为`price1;price2;price3`， 其中`;`为序列分隔符；支持ARRAY<float>为`[price1,price2,price3]`或者ARRAY\<ARRAY<float>>类型为`[[emb11,emb12],[emb21,emb22]]`（建议，性能更好）

```
feature_configs: {
    sequence_raw_feature {
        feature_name: "click_price_seq"
        sequence_length: 50
        sequence_delim: ";"
        expression: "user:click_price_seq"
    }
}
```

- **sequence_length**: 序列特征最大长度
- **sequence_delim**: 序列特征分隔符
- **expression**: 特征FG所依赖的字段来源，由两部分组成`input_side`:`input_name`
- 其余配置同RawFeature

## SequenceFeature：分组序列特征

分组序列特征的子序列格式一般为`XX;XX;XX`，如用户点击的Item的序列特征为`item_id1;item_id2;item_id3`，其中`;`为序列分隔符，也支持ARRAY类型，同SequenceIdFeature和SequenceRawFeature。

分组序列特征支持使用物品或行为的属性构建一组子序列，如类目序列`cate1;cate2;cate1`、品牌序列`brand1;brand2;brand1`、行为时间序列`ts1;ts2;ts3`等。一条样本中，同一分组的子序列的长度需要保持相同。

分组序列特征在线上模型服务时比其他序列特征更加高效，只需传递`sequence_pk`，线上模型服务从物品特征内存Cache中关联出物品属性子特征的序列，无需从请求中传递。

```
feature_configs: {
    sequence_feature {
        sequence_name: "click_seq"
        sequence_length: 50
        sequence_delim: ";"
        sequence_pk: "user:click_seq_pk"
        features {
            id_feature {
                feature_name: "item_id"
                expression: "item:iid"
                embedding_dim: 32
                hash_bucket_size: 100000
            }
        }
        features {
            id_feature {
                feature_name: "cate"
                expression: "item:cate"
                embedding_dim: 32
                hash_bucket_size: 1000
            }
        }
        features {
            raw_feature {
                feature_name: "ts"
                expression: "user:ts"
            }
        }
    }
}
```

- **sequence_name**: 序列特征名
- **sequence_length**: 序列特征最大长度
- **sequence_delim**: 序列特征分隔符
- **sequence_pk**: 序列特征主键，一般为ItemId列表，主要用于线上模型服务，线上模型服务会使用该ItemId列表从物品特征内存Cache中关联出物品属性子特征的序列，无需从请求中传递。而行为属性相关子序列（如行为时间序列`ts1;ts2;ts3`）跟用户相关，则仍需从请求从传递。
- **features**: 序列特征子特征，配置同IdFeature和RawFeature
  - **feature_name**: 子特征特征名，完整的子特征名应拼接上`${sequence_name}__`前缀，以上述配置中`item_id`子特征为例，子特征名列名应为`click_seq__item_id`
  - **expression**: 特征FG所依赖子特征字段来源名，由两部分组成`input_side`:`input_name`。在输入样本数据中列名应拼接上`${sequence_name}__`前缀，以上述配置中`item_id`子特征为例，`expression`为`item:iid`，输入样本数据中列名应为`click_seq__iid`。在线上模型服务中，如果子特征的`input_side`为`item`，子序列无需从请求中传递；如果子特征的`input_side`为`user`，子序列需要从请求中传递。
  - 其中当类型为IdFeature时
    - **value_dim**: 目前只支持value_dim=1，不支持多值ID序列
