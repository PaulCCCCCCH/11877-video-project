# Notes

## `models/networks.py`
- model01: input channel (3) -> 4ngf + norm + ReLU
- model02: 4ngf -> 8ngf + norm + ReLU
- model03: 
    - 8ngf -> 12ngf + norm + ReLU
    - 12ngf -> 16ngf + norm + ReLU
- model04: 
    - 16ngf -> 16ngf + norm + ReLU + max pool + norm + ReLU
    - 16ngf -> 12ngf
- model2 (decoder):
    -   12ngf -> 16ngf -> 16ngf -> 16ngf -> 12ngf -> 8ngf -> 4ngf -> ngf -> output channel (3)
