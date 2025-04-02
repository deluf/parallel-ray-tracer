class Skybox{
  PImage img;
  boolean exists;
  
  Skybox(PImage img){
    this.img = img.copy();
    this.img.loadPixels();
    exists = true;
  }
  
  Skybox(boolean exists){
    this.exists = exists;
  }
  
  float[] getColor(Vector dir){
    float[] out = new float[3];
    //WRAP IMAGE IN SPHERE AND CENTER IT
    float u = 0.75 - atan2(dir.y, dir.x)/(2*PI);
    float v = 0.5 - asin(dir.z)/PI;
    
    //BILINEAR INTERPOLATION
    float valX = u*img.width - int(u*img.width);
    float valY = v*img.height - int(v*img.height);
    
    int x = int(u*img.width) % img.width;
    int y = int(v*img.height) % img.height;
    
    int pixel00 = img.pixels[x + y*img.width];
    int pixel10 = img.pixels[((x+1) % img.width) + y*img.width];
    int pixel01 = img.pixels[x + ((y+1) % img.height)*img.width];
    int pixel11 = img.pixels[((x+1) % img.width) + ((y+1) % img.height)*img.width];
    
    float r = bilinear((pixel00 & 0xFF0000) >> 16, (pixel10 & 0xFF0000) >> 16, (pixel01 & 0xFF0000) >> 16, (pixel11 & 0xFF0000) >> 16, valX, valY);
    float g = bilinear((pixel00 & 0xFF00) >> 8, (pixel10 & 0xFF00) >> 8, (pixel01 & 0xFF00) >> 8, (pixel11 & 0xFF00) >> 8, valX, valY);
    float b = bilinear((pixel00 & 0xFF), (pixel10 & 0xFF), (pixel01 & 0xFF), (pixel11 & 0xFF), valX, valY);
    
    out[0] = r / 255.0;
    out[1] = g / 255.0;
    out[2] = b / 255.0;
    return out;
  }
}

Skybox loadSkybox(String file){
  Skybox sky = new Skybox(false);
  String[] s = loadStrings(file+".obj");
  if(s.length == 1){
    PImage img = loadImage(dir+"/"+s[0]);
    sky = new Skybox(img);
  }
  return sky;
}
