from lib import *
import gpytorch
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


### GP Model

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def pprint(*args, end='\n'):
    print(*args, end=end)
    with open('../logs/'+common_path+'.txt', 'a') as f:
        f.write(' '.join(map(str, args)))
        f.write(end)

# Config.res_path = Config.res_path.replace('/workspace', '/home')
# m_name = sys.argv[1]
# c_fold = sys.argv[2]
# sampling = sys.argv[3]
# Xcols = sys.argv[4]
# kernel = sys.argv[5]
# timekernel = sys.argv[6]

m_name = 'nsgp'
c_fold = '2'
sampling = 'cont' # cont, nn, uni
Xcols = '@'.join(['longitude', 'latitude', 'humidity', 'temperature', 'weather', 'wind_direction', 'wind_speed', 'delta_t'])
kernel = 'rbf' # Order: RBF, M32
timekernel = 'loc_periodic' # Order RBF, loc_per


def test(epoch):
    if len(Xcols.split('@'))>15:
        common_path = 'one@hot@encoded'+c_fold
    else:
        common_path = '_@_'.join([m_name, c_fold, Xcols, sampling, kernel, timekernel])
    train_res = pd.read_pickle(Config.res_path+common_path+'_Epoch' + str(epoch) +'.res')

    dataloader = train_res['dataloader']
    test_X, test_y = dataloader.load_test('target')
    X, y, _ = dataloader.load_train()

    model = torch.load(Config.res_path+common_path+'_Epoch'+str(epoch)+'.model')
    model.eval()

    if m_name in ['nsgp', 'snsgp']:
        with torch.no_grad():
            pred_y, pred_var = model.predict(X.to(Config.device), y.to(Config.device),
                                            test_X.to(Config.device))
    else:
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(Config.device)
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_X.to(Config.device)))
            pred_y = observed_pred.mean
            pred_var = observed_pred.variance
            print(list(model.parameters()))

    dataloader.test_data['pred_mean'] = pred_y.cpu().ravel() + dataloader.y_mean
    if m_name in ['nsgp', 'snsgp']:
        dataloader.test_data['pred_var'] = pred_var.diagonal().cpu()
    else:
        dataloader.test_data['pred_var'] = pred_var.cpu()
    dataloader.test_data.to_csv(Config.res_path+common_path+'_Epoch'+str(epoch)+'.csv')
    # pprint('Finished')

def fold_wise_rmse(fold):
    test_data = pd.read_csv(f_name(fold))
    # print('log:', test_data['pred_mean'].shape, test_data['pred_mean'].dropna().shape)
    # print('loss', sum(pd.read_pickle(f_name_res(fold))['loss'][-5:-1])/4)
    test_data =test_data.dropna()
    for sid in test_data.station_id.unique():
        tmp_df = test_data[test_data.station_id==sid]
        tmp_df = tmp_df.drop(tmp_df.index[(tmp_df['filled']==True)])
        # print(sid, mean_squared_error(tmp_df['PM25_Concentration'], tmp_df['pred_mean'], squared=False))
    return 'fold', fold, 'RMSE', mean_squared_error(test_data['NO2_Concentration'], test_data['pred_mean'], squared=False)

def fold_wise_mae(fold):
    test_data = pd.read_csv(f_name(fold))
    # print('log:', test_data['pred_mean'].shape, test_data['pred_mean'].dropna().shape)
    # print('loss', sum(pd.read_pickle(f_name_res(fold))['loss'][-5:-1])/4)
    test_data =test_data.dropna()
    for sid in test_data.station_id.unique():
        tmp_df = test_data[test_data.station_id==sid]
        tmp_df = tmp_df.drop(tmp_df.index[(tmp_df['filled']==True)])
        # print(sid, mean_absolute_error(tmp_df['PM25_Concentration'], tmp_df['pred_mean']))
    return 'fold', fold, 'MAE', mean_absolute_error(test_data['NO2_Concentration'], test_data['pred_mean'])

def fold_wise_mape(fold):
    test_data = pd.read_csv(f_name(fold))
    # print('log:', test_data['pred_mean'].shape, test_data['pred_mean'].dropna().shape)
    # print('loss', sum(pd.read_pickle(f_name_res(fold))['loss'][-5:-1])/4)
    test_data =test_data.dropna()
    for sid in test_data.station_id.unique():
        tmp_df = test_data[test_data.station_id==sid]
        tmp_df = tmp_df.drop(tmp_df.index[(tmp_df['filled']==True)])
        # print(sid, mean_absolute_error(tmp_df['PM25_Concentration'], tmp_df['pred_mean']))
    return 'fold', fold, 'MAPE', mean_absolute_percentage_error(test_data['NO2_Concentration'], test_data['pred_mean'])

best_rmse = float('inf')
best_mae = float('inf')
best_mape = float('inf')

for i in range(10, 300, 10):

    common_path = lambda c_fold: '_@_'.join([m_name, c_fold, Xcols, sampling, kernel, timekernel])
    f_name = lambda fold: Config.res_path + common_path(fold) + '_Epoch' + str(i) + '.csv'

    try:
        test(i)
        f1_rmse = fold_wise_rmse(c_fold)[3]
        f1_mae = fold_wise_mae(c_fold)[3]
        f1_mape = fold_wise_mape(c_fold)[3]
    except Exception as e:
        print(e)
        continue

    if best_rmse > f1_rmse:
        best_rmse = f1_rmse
        best_mae = f1_mae
        best_mape = f1_mape

print('RMSE: %f, MAE: %f, MAPE: %f' % (best_rmse, best_mae, best_mape))
