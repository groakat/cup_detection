from ..ml import cardboard as MLCB
import pylab as plt


def plot_frame_predictions(prediction_d, show_smoothing=False):
    gt = prediction_d['ground-truth']
    pred = prediction_d['predict']
    frame_nums = range(len(pred))

    # quick fix that prevents creation of strange polygons
    if gt[-1] == 0:
        gt[-1] = 1

    plt.fill(frame_nums, gt, facecolor='r', alpha=0.5)
    plt.plot(frame_nums, pred * 0.6, 'b')

    if show_smoothing:
        y_smooth = MLCB.apply_smoothing(pred)
        plt.plot(frame_nums, y_smooth['predict'] * 0.8, 'DarkRed', linewidth=2)
