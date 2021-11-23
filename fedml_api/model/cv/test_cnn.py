from ptflops import get_model_complexity_info

from fedml_api.model.cv.cnn import CNNDropOut

if __name__ == "__main__":
    # net = CNN_OriginalFedAvg()
    net = CNNDropOut()

    flops, params = get_model_complexity_info(net, (1, 28, 28), as_strings=True,
                                              print_per_layer_stat=True, verbose=True)
    print(params)
    print(f'{"Computational complexity: ":<30}  {flops:<8}')
    print(f'{"Number of parameters: ":<30}  {params:<8}')
