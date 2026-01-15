# Demystifying the Slash Pattern in Attention: The Role of RoPE
Yuan Cheng, Fengzhuo Zhang, Yunlong Hou などによる
軽く読んだ程度

LLMのAttentionにおいて特定の相対位置に注意が集中する「スラッシュパターン（Slash Patterns）」がなぜ発生するのかについて、RoPEの影響を理論的に論じた論文
LLMの埋め込み空間においてKey, Queryが円錐状に分布しており、その結果RoPE適用前のKey, Query行列はほぼランク1であることを発見
これにより、Attentionスコアの変動は主にRoPEが支配的であり、これがスラッシュパターンを自然に起こすと結論づけている
