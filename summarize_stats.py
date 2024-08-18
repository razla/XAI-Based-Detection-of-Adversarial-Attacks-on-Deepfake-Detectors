import argparse
import json
import glob
import os
import pandas as pd


def summarize_videos(work_dir=None, model_name='xception', majority_vote_threshold=0.5):
    table_columns = ['file_name', 'detection_model', 'total_fake_frames', 'total_real_frames', 'percent_fake_frame',
                     'percent_real_frame', 'detector_classification', 'ground_truth_classification']
    summery_table = pd.DataFrame(columns=table_columns)
    row = dict.fromkeys(table_columns, None)
    json_files = glob.glob(f'{os.path.join(work_dir, model_name)}/*.json')
    for json_path in json_files:
        with open(json_path) as f:
            file_name = os.path.basename(json_path)
            data = json.load(f)
            row['file_name'] = file_name
            row['detection_model'] = model_name
            row['total_fake_frames'] = data['total_fake_frames']
            row['total_real_frames'] = data['total_real_frames']
            row['percent_fake_frame'] = data['percent_fake_frames']
            row['percent_real_frame'] = 1 - data['percent_fake_frames']
            row['detector_classification'] = 'fake' if data[
                                                           'percent_fake_frames'] > majority_vote_threshold else 'real'
            summery_table = pd.concat([summery_table, pd.DataFrame([row])], ignore_index=True)
    stats = {model_name: calc_videos_stats(df=summery_table, model_name=model_name)}
    return summery_table, stats


def summarize_frames(work_dir=None, model_name='xception', majority_vote_threshold=0.5):
    table_columns = ['file_name', 'frame_id', 'detection_model', 'detector_classification',
                     'ground_truth_classification',
                     'percent_real', 'percent_fake']
    row = dict.fromkeys(table_columns, None)
    summery_table_row_list = []
    json_files = glob.glob(f'{os.path.join(work_dir, model_name)}/*.json')
    if json_files:
        for json_path in json_files:
            with open(json_path) as f:
                file_name = os.path.basename(json_path)
                data = json.load(f)
                for i, frame in enumerate(data['probs_list']):
                    row['file_name'] = file_name
                    row["frame_id"] = i
                    row['detection_model'] = model_name
                    row['detector_classification'] = 'fake' if frame.index(max(frame)) == 1 else 'real'
                    row['detector_classification'] = 'fake' if frame[1] > majority_vote_threshold else 'real'
                    row['percent_real'] = frame[0]
                    row['percent_fake'] = frame[1]
                    summery_table_row_list.append(row.copy())
    summery_table = pd.DataFrame.from_records(summery_table_row_list)
    stats = {model_name: calc_frames_stats(df=summery_table, model_name=model_name)}
    return summery_table, stats


def calc_videos_stats(df, model_name=None):
    assert model_name is not None, "must input model name"
    model_rows = df.loc[df.detection_model == model_name]
    mean_real_frames_percent = model_rows.percent_real_frame.mean()
    mean_fake_frames_percent = model_rows.percent_fake_frame.mean()
    mean_total_frame = (model_rows.total_real_frames + model_rows.total_fake_frames).mean()
    fake_count = (model_rows.detector_classification == 'fake').sum()
    real_count = (model_rows.detector_classification == 'real').sum()
    fake_precision = fake_count / model_rows.shape[0]
    real_precision = real_count / model_rows.shape[0]
    total_fake_vids = (model_rows.ground_truth_classification == 'fake').sum()
    total_real_vids = (model_rows.ground_truth_classification == 'real').sum()
    total_dict = {"mean_real_frames_percent": mean_real_frames_percent,
                  "mean_fake_frames_percent": mean_fake_frames_percent,
                  "mean_total_frame": mean_total_frame,
                  "detector_fake_count": fake_count,
                  "detector_real_count": real_count,
                  "detector_fake_precision": fake_precision,
                  "detector_real_precision": real_precision,
                  "total_fake_videos": total_fake_vids,
                  "total_real_videos": total_real_vids
                  }
    return pd.Series(data=total_dict)


def calc_frames_stats(df, model_name=None):
    assert model_name is not None, "must input model name"
    model_rows = df.loc[df.detection_model == model_name]
    mean_real_frame_percent = model_rows.percent_real.mean()
    mean_fake_frame_percent = model_rows.percent_fake.mean()
    fake_count = (model_rows.detector_classification == 'fake').sum()
    real_count = (model_rows.detector_classification == 'real').sum()
    model_precision_real = real_count / len(model_rows)
    model_precision_fake = fake_count / len(model_rows)

    total_dict = {"mean_real_frame_percent": mean_real_frame_percent,
                  "mean_fake_frame_percent": mean_fake_frame_percent,
                  "detector_fake_count": fake_count,
                  "detector_real_count": real_count,
                  "model_precision_real": model_precision_real,
                  "model_precision_fake": model_precision_fake,
                  }
    return pd.Series(data=total_dict)


def calculate_accuracy(df, threshold):
    from sklearn.metrics import confusion_matrix
    dfcopy = df.copy()
    dfcopy.loc[dfcopy['percent_real'] < threshold, 'detector_classification'] = 0
    dfcopy.loc[dfcopy['percent_real'] >= threshold, 'detector_classification'] = 1
    dfcopy.loc[dfcopy['ground_truth_classification'] == 'real', 'ground_truth_classification'] = 0
    dfcopy.loc[dfcopy['ground_truth_classification'] == 'fake', 'ground_truth_classification'] = 1
    true_labels = dfcopy['ground_truth_classification'].to_list()
    predicted_labels = dfcopy['detector_classification'].to_list()
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    accuracy = (tn + tp) / (tn + tp + fn + tp)
    return accuracy


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset_path', type=str)
    p.add_argument('--model_type', type=str, default='xception')
    p.add_argument('--threshold', type=float, default=0.5)
    args = p.parse_args()

    dataset_path = args.dataset_path
    model_type = args.model_type
    threshold = args.threshold

    majority_vote_threshold = args.threshold
    videos_dataset_summ, videos_stats = summarize_videos(work_dir=dataset_path, model_name=model_type,
                                                         majority_vote_threshold=majority_vote_threshold)
    frames_dataset_summ, frames_stats = summarize_frames(work_dir=dataset_path, model_name=model_type,
                                                         majority_vote_threshold=majority_vote_threshold)

    print("============Attacked Videos Dataset stats============")
    print(f"====={model_type}=====")
    print(videos_stats[model_type])
    print("\n")
    print("============Attacked Frames Dataset stats============")
    print(f"====={model_type}=====")
    print(frames_stats[model_type])
    print("\n\n")
    pass
