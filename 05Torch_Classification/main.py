import myModel
import train


train_dl, valid_dl = train.get_data(myModel.train_ds, myModel.valid_ds, myModel.bs)
model, opt = train.get_model()
train.fit(25, model, myModel.loss_func, opt, train_dl, valid_dl)
