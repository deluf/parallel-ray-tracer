class RayTracer extends Thread {
  int from_x, to_x;
  int from_y, to_y;
  Vector ul;
  Vector origin;
  Vector incX, incY;
  
  RayTracer(int from_x, int to_x, int from_y, int to_y, Vector origin, Vector ul, Vector incX, Vector incY) {
    this.from_x = from_x;
    this.to_x = to_x;
    this.from_y = from_y;
    this.to_y = to_y;
    this.origin = origin;
    this.ul = ul;
    this.incX = incX;
    this.incY = incY;
  }

  public void run() {
    for(int y = from_y; y < to_y; y++){
      for(int x = from_x; x < to_x; x++){
        Vector dir = Vector.sub(ul, origin);
        dir.add( Vector.mult(incX, x) );
        dir.add( Vector.mult(incY, y) );
        //check sphere intersection
        float[] col = new float[3];
        for(int ssaaY = 1; ssaaY <= SSAA; ssaaY++){
          for(int ssaaX = 1; ssaaX <= SSAA; ssaaX++){
             Vector sample = dir.copy(); 
             sample.add(Vector.mult(incX, ssaaX/(SSAA+1.0)));
             sample.add(Vector.mult(incY, ssaaY/(SSAA+1.0))); 
             sample.normalize();
             float[] out = rayTrace(origin, sample, 0);
             for(int i = 0; i < 3; i++)
               col[i] += out[i];
          }
        }
        for(int i = 0; i < 3; i++)
          col[i] = col[i]/(SSAA*SSAA);
        pixels[x + y*width] = (255<<24) | (int(col[0]*255)<<16) | (int(col[1]*255)<<8) | int(col[2]*255);
      }
    }
    return;
  }
  
}
