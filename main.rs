use csv::ReaderBuilder;
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::{s, Array2, Array3, Axis};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use show_image::{create_window, ImageInfo, ImageView};
use std::fs::File;
use std::convert::TryInto;
use tch::{nn, nn::Module, nn::OptimizerConfig, Tensor, Device};

//-----------------------
// LOADING CSV DATA
//-----------------------
fn load_csv_data(file_path: &str) -> Array2<f32> {
    let file = File::open(file_path).expect("Failed to open file");
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);

    let mut records = vec![];
    for result in reader.records() {
        let record = result.expect("Failed to read record");
        let row: Vec<f32> = record.iter().map(|s| s.parse().unwrap()).collect();
        records.push(row);
    }

    let array: Array2<f32> = Array2::from_shape_vec((records.len(), records[0].len()), records.into_iter().flatten().collect()).unwrap();

    array
}

//-----------------------
// NORMALIZE
//-----------------------
fn normalize(tensor: &mut Array2<f32>, mean: &[f32], std: &[f32]) {
    for (i, (mean, std)) in mean.iter().zip(std.iter()).enumerate() {
        tensor.index_axis_mut(Axis(1), i).iter_mut().for_each(|x| *x = (*x - mean) / std);
    }
}

//-----------------------
// RANDOM HORIZONTAL FLIPPING
//-----------------------
fn random_horizontal_flip(tensor: &mut Array2<f32>, probability: f64, img_width: usize) {
    let mut rng = rand::thread_rng();
    if rng.gen_bool(probability) {
        for i in 0..3 {
            let mut img_slice = tensor.slice_mut(s![.., i * img_width..(i + 1) * img_width]);
            img_slice.invert_axis(Axis(1));
        }
    }
}

//----------------------------
// ADD GAUSSIAN NOISE 
//----------------------------
fn add_gaussian_noise(tensor: &mut Array2<f32>, mean: f32, std: f32) {
    let normal = Normal::new(mean, std).unwrap();
    let mut rng = rand::thread_rng();
    for val in tensor.iter_mut() {
        *val += normal.sample(&mut rng);
    }
}

//-----------------------
// CONVERTING IMAGE TO TENSOR
//-----------------------
fn image_to_tensor(img: &DynamicImage) -> Array3<f32> {
    let (width, height) = img.dimensions();
    let mut tensor = Array3::<f32>::zeros((3, height as usize, width as usize));
    for (x, y, pixel) in img.pixels() {
        tensor[(0, y as usize, x as usize)] = pixel[0] as f32 / 255.0;
        tensor[(1, y as usize, x as usize)] = pixel[1] as f32 / 255.0;
        tensor[(2, y as usize, x as usize)] = pixel[2] as f32 / 255.0;
    }
    tensor
}

//-----------------------
// SHOW IMAGE FROM ARRAY
//-----------------------
fn show_image_from_array(data: &Array2<f32>, width: u32, height: u32) {
    let flat_data: Vec<u8> = data.iter().map(|x| (*x * 255.0) as u8).collect();
    let image_buffer = image::RgbImage::from_raw(width, height, flat_data).expect("Failed to create image buffer");

    let image_info = ImageInfo::rgb8(width, height);
    let image_view = ImageView::new(image_info, &image_buffer);
    let window = create_window("image", Default::default()).unwrap();
    window.set_image("image", image_view).unwrap();
}

//-----------------------
// MAIN FUNCTION
//-----------------------
fn main() {
    let csv_file_path = "/Users/aryanmishra/Desktop/DSA/Imagenet16_train/train_data_combined.csv";
    let img_width = 32;
    let img_height = 32;
    let channels = 3;
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    let augment = true;

    let mut data = load_csv_data(csv_file_path);
    println!("Loaded data with shape: {:?}", data.shape());

    println!("Sample data (first row): {:?}", data.row(0));

    normalize(&mut data, &mean, &std);
    println!("Data normalized.");

    if augment {
        random_horizontal_flip(&mut data, 0.5, img_width);
        println!("Random horizontal flip applied.");
        // add_gaussian_noise(&mut data, 0.0, 0.1);
        // println!("Gaussian noise added.");
    }

    println!("Preparing to reshape the sample image.");
    let sample_image_row = data.row(0).to_owned();
    println!("Sample row extracted: {:?}", sample_image_row);
    let sample_image: Array2<f32> = sample_image_row.into_shape((img_height * img_width, channels)).unwrap();
    println!("Sample image reshaped.");

    println!("Displaying the image.");
    show_image_from_array(&sample_image, img_width.try_into().unwrap(), img_height.try_into().unwrap());
    println!("Image displayed.");
}
