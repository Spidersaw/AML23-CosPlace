import matplotlib.pyplot as plt


def get_count(classes, images):
    #return list containing [class_index]*no.instances per class
    #[0,0,0,0...,1,1,1,...,2,2,2,2....]
    count = []
    for i in range(int(len(classes))):
      count.extend([i]*len(images[classes[i]]))
    return count

#Histogram showing how many cells contain a given number of images. (Density?)


#Histogram showing number of images per class

#classes is an array of tuples (UTM_east, UTM_north, heading) = class_id
#images is a dict in which for each class_id, there is an array of images paths
def plot_histogram(classes, images):
    #x = classes
    #y = num. of images

    count = get_count(classes,images)
        
    plt.figure()
    plt.title("Number of images per class")
    plt.xlabel("Classes")
    plt.ylabel("Count")
    
    plt.hist(count, bins=range(int(len(classes))+1), alpha=1)
    plt.savefig('%s/%s.jpeg' % ('/content','no_images_per_class'))
    plt.show()

