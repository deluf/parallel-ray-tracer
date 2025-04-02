class Triangle{
  Vector[] coords;
  //reflection power
  float[] ks;
  //absorption power
  float[] kd;
  // raytracing power
  float[] kr;
  
  Vector[] norm = new Vector[2];
  
  Triangle(Vector a, Vector b, Vector c, float[] ks, float[] kd, float[] kr){
    coords = new Vector[3];
    this.ks = ks.clone();
    this.kd = kd.clone();
    this.kr = kr.clone();
    coords[0] = a.copy();
    coords[1] = b.copy();
    coords[2] = c.copy();
    
    //calculate norm
    Vector e1 = Vector.sub(coords[1], coords[0]);
    Vector e2 = Vector.sub(coords[2], coords[0]);
    norm[0] = e1.cross(e2);
    norm[1] = e2.cross(e1);
  }
}

Triangle[] loadTriangles(String file){
  String[] mtl = loadStrings(file+".mtl");
  String[] s = loadStrings(file+".obj");
  float[] ks = new float[3];
  float[] kd = new float[3];
  float[] kr = new float[3];
  int count = 0;
  for(int i = 0; i < s.length; i++){
    if(s[i].charAt(0) == 'v')
      count++;
  }
  Vector[] vertices = new Vector[count];
  count = 0;
  for(int i = 0; i < s.length; i++){
    if(s[i].charAt(0) == 'v'){
      float[] data = float(split(s[i], ' '));
      vertices[count] = new Vector(data[1], data[2], data[3]);
      count++;
    }
  }
  count = 0;
  for(int i = 0; i < s.length; i++){
    if(s[i].charAt(0) == 'f'){
      count++;
    }
  }
  Triangle[] triangles = new Triangle[count];
  count = 0;
  for(int i = 0; i < s.length; i++){
    if(s[i].length() >= 6 && s[i].substring(0, 6).equals("usemtl") ){
      String material = s[i].substring(7);
      for(int j = 0; j < mtl.length; j++){
        if(mtl[j].length() == 7+material.length() && mtl[j].substring(7).equals(material)){
          kd = float(split(mtl[j+3].substring(3), ' ')).clone();
          ks = float(split(mtl[j+4].substring(3), ' ')).clone();
          kr = float(split(mtl[j+5].substring(3), ' ')).clone();
        } 
      }
    }
    if(s[i].charAt(0) == 'f'){
      int[] data = int(split(s[i], ' '));
      triangles[count] = new Triangle(vertices[data[1]-1], vertices[data[2]-1], vertices[data[3]-1], ks, kd, kr); 
      count++;
    }
  }
  return triangles;
}
