import argparse
import numerapi
import os
import pandas as pd
from halo import Halo
from sklearn.ensemble import GradientBoostingRegressor

TARGET_NAME = 'target_20d'
PREDICTION_NAME = 'signal'

spinner = Halo(text='', spinner='dots')


def main(output_dir=None):
    """Creates example_signal_yahoo.csv to upload for validation and live data submission"""

    spinner.start('Reading data')
    train = pd.read_csv('example_training_data_yahoo.csv')
    tournament = pd.read_csv('tournament_data_yahoo.csv')
    spinner.succeed()

    feature_names = train.filter(like='feature_').columns.to_list()

    spinner.start('Training model')
    model = GradientBoostingRegressor(subsample=0.1)
    model.fit(train[feature_names], train[TARGET_NAME])
    spinner.succeed()

    # predict test and live data
    spinner.start('Predicting test and live data')

    # drop rows where target or features are null
    tournament = tournament.dropna(subset=feature_names)
    tournament[PREDICTION_NAME] = model.predict(tournament[feature_names])
    spinner.succeed()

    # prepare and writeout example file
    spinner.start('Writing signal upload file')
    diagnostic_df = tournament.copy()
    diagnostic_df['data_type'] = diagnostic_df.data_type.fillna('live')

    example_signal_output_path = 'example_signal_upload.csv'
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        example_signal_output_path = f'{output_dir}/{example_signal_output_path}'

    diagnostic_df = diagnostic_df.drop_duplicates(subset=["bloomberg_ticker", "friday_date"], keep="first")
    diagnostic_df[['bloomberg_ticker', 'friday_date', 'data_type', 'signal']].reset_index(drop=True).to_csv(example_signal_output_path, index=False)
    spinner.succeed()
    
    spinner.start("Submitting Signal predictions")
    spinner.succeed()
    
    SubmitSignal('janky')
    SubmitSignal('danky')
    SubmitSignal('hanky')
    SubmitSignal('panky')
    SubmitSignal('cranky')
    SubmitSignal('blanky')
    SubmitSignal('wanky')
    SubmitSignal('yanky')
    SubmitSignal('lanky')
    SubmitSignal('ranky')
    SubmitSignal('pee')
    SubmitSignal('poo')
    SubmitSignal('poop')
    SubmitSignal('pleep')
    SubmitSignal('ploop')
    SubmitSignal('diarrhea')
    SubmitSignal('coffeesquirts')
    SubmitSignal('farty')
    SubmitSignal('bumbum')
    SubmitSignal('leftbum')
    SubmitSignal('rightbum')
    SubmitSignal('topbum')
    SubmitSignal('bottombum')
    SubmitSignal('frontbum')
    SubmitSignal('backbum')
    SubmitSignal('centerbum')

def SubmitSignal(accountName):
    public_key = "HSMNVJAP7LXDV5HVSLDHXMEIHALH2UU4"
    private_key = "PVVU5EUHET5K6K5GIRVBMFYIO43WZRB45QSUWQLWFCSTOLG3JCPBCUPWUO4RYOX6"
    napi = numerapi.SignalsAPI(public_key, private_key)
    model_id = napi.get_models()[f'{accountName}']
   
    submission = napi.upload_predictions("example_signal_upload.csv", model_id=model_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Signals example data pipeline')
    parser.add_argument('--output_dir', default=None)

    args = parser.parse_args()
    main(args.output_dir)
