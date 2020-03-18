from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_three_views(filename, points, titles, suptitle='', sizes=None,
                     cmap='Reds', zdir='y', xlim=(-0.3, 0.3), ylim=(-0.3, 0.3),
                     zlim=(-0.3, 0.3)):
    if sizes is None:
        sizes = [0.5 for i in range(len(points))]

    fig = plt.figure(figsize=(len(points) * 3, 9))

    for i in range(3):
        elev = 30
        azim = -45 + 90 * i
        for j, (points, size) in enumerate(zip(points, sizes)):
            color = points[:, 0]
            ax = fig.add_subplot(3, len(points), i * len(points) + j + 1,
                                 projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], zdir=zdir,
                       c=color, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)


def visualize_cloud_points(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


def plot_cloud_points(ax, points):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], zdir='y',
               c=points[:, 0], s=0.5, cmap='Reds', vmin=-1, vmax=0.5)
    ax.set_axis_off()
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.3, 0.3)


def plot_side_by_side(partial, complete):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(121, projection='3d')
    plot_cloud_points(ax, partial)
    ax.set_title('Partial cloud')
    ax = fig.add_subplot(122, projection='3d')
    plot_cloud_points(ax, complete)
    ax.set_title('Complete cloud')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
