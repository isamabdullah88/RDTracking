
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Visualize speed, orientation for each id tracker

def visualize(trackers, tracker_ids, show=True):
    """
    Visualization system for trackers. Visualizes speed and orientation for each unique  tracker.
    """
    num_trackers = len(trackers)
    xl, xr = (42.499606417905724, 42.499868)
    yl, yr = (90.696186, 90.69658661844755)

    gs = gridspec.GridSpec(2, num_trackers)
    fig = plt.figure(figsize=(12,10))

    ax_t = fig.add_subplot(gs[1,:])
    ax_t.set_title('Tracking')
    ax_t.set_xlabel('x')
    ax_t.set_ylabel('z')
    ax_t.set_xlim(xl, xr)
    ax_t.set_ylim(yl, yr)

    xs = []
    ys = []

    print('tracker id: '); print(tracker_ids)
    for i,tracker_idx in enumerate(tracker_ids):
        tracker_id = tracker_ids[tracker_idx]
        tracker = trackers[tracker_idx]
        x, y, _, _, v, theta = tracker.get_state()

        xs.append(x)
        ys.append(y)

        ax = fig.add_subplot(gs[0, i])
        ax.set_title('Tracker: {0:d}'.format(tracker_id))
        ax.set_ylabel('speed')
        ax.plot(v, '^');
        # ax.plot(theta, '.')
        
        ax_t.plot(x, y, '*')
        ax_t.text(x, y, str(tracker_id))

    

    if show:
        plt.show()



def main():
    trackers = [1,2,3,4,5]
    visualize(trackers)


if __name__ == "__main__":
    main()
