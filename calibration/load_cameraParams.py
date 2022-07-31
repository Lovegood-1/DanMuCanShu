from pyrsistent import v
import scipy.io as scio
import os
import pdb

def load_mat(params_path='cameraParams_new.mat', params_name='cameraParams_new'):
    # pdb.set_trace()
    # 载入.mat相机参数 以字典形式存储
    path_mat = os.path.join(params_path)
    cameraParams = scio.loadmat(path_mat)[params_name]
    # a = cameraParams.dtype
    params_list = ['ImageSize', 'RadialDistortion', 'TangentialDistortion', 'WorldPoints', 'WorldUnits', \
                'EstimateSkew', 'NumRadialDistortionCoefficients', 'EstimateTangentialDistortion', \
                'TranslationVectors', 'ReprojectionErrors', 'RotationVectors', 'NumPatterns', 'IntrinsicMatrix', \
                'FocalLength', 'PrincipalPoint', 'Skew', 'MeanReprojectionError', 'ReprojectedPoints', 'RotationMatrices']
    params_value = [cameraParams[param_item] for param_item in params_list]
    params = dict(zip(params_list, params_value))
    for param_item in params:
        print(param_item)
        # print(param_item, params[param_item])

    return params

if __name__ == "__main__":
    params_path=r'data\0727\cameraParams_new0727.mat'
    params_name='cameraParams_new'

    params = load_mat(params_path, params_name)
    a = 1