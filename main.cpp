//the program was created on september 25th, 2017 by zzc
//it complishes the function about intelligent scissors
//the input is a picture with .JPG and not too large
#include<opencv2/opencv.hpp>  
#include<opencv2/highgui/highgui.hpp>  
#include<opencv2/imgproc/imgproc.hpp>  
#include<cmath>
#include<iostream>
#include <queue>
using namespace cv; 
using namespace std;
const double pi=3.1415926;
const long long ex=1e6;
const int inf=10000000;
const double Wd=0.43,Wz=0.43,Wg=0.14;
const int maxn=1e6;
Mat lapl,lapl2,gray,mag,gau,dir,lap_plus_gauss,result,src; 
double gx[1000][1000][8],gy[1000][1000][8];
double magx[1000][1000],magy[1000][1000],_mag[1000][1000],lap[1000][1000];
int x[]={1,-1,0,2,-2};
int y[]={1,-1,0,2,-2};
int directionx[]={-1,0,1,-1,1,-1,0,1},directiony[]={-1,-1,-1,0,0,1,1,1};
int pre[maxn];
double last[maxn];
int vis[maxn];
int startx,starty,endx,endy,flag;
uchar Laplacian(int x,int y);
uchar Magnitude(int x1,int y1);
uchar Gaussian(int x0,int y0);
uchar Laplacian2(int x,int y);
double stmag(int x2,int y2);
void cal_L(int i,int j);
void cal_dp(int i,int j);
void cal_mag(int i,int j);
void cal_fd(int i,int j);
void cal_value(int i,int j);
void init();
void draw(int y0,int x0);
void dij(int width,int height,int startx,int starty,int endx,int endy);
void on_mouse( int event, int x, int y, int flags, void* ustc);
void on_mouse_click( int event, int x, int y, int flags, void* ustc) ;
struct NN{
	int v;double t;
	bool operator<(const NN& tt)const
	{
		return t<tt.t;
	}
};
struct heap{
	NN num[10000];
	int size;
	void build(){
		size=0;
	}
	void push(NN t){
		assert(size<10000);
		num[size++]=t;
		up(size-1);
	}
	NN pop(){
		NN r=num[0];
		num[0]=num[size-1];
		size--;
		down(0);
		return r;
	}
	void down(int index){
		NN t=num[index];
		index=index*2+1;
		while(index<size){
			if(index+1<size&&num[index+1]<num[index])index++;
			if(t<num[index])break;
			else{
				num[(index-1)/2]=num[index];
				index=index*2+1;
			}
		}
		num[(index-1)/2]=t;
	}
	void up(int index){
		NN t=num[index];
		while(index>0&&t<num[(index-1)/2]){
			num[index]=num[(index-1)/2];
			index=(index-1)/2;
		}
		num[index]=t;
	}
};
int main( )  
{     
	int tt;
	init();
	//cin>>tt;
    src = imread("6.jpg");  //there woulde be a picture under the project file
    cvtColor( src, gray, CV_BGR2GRAY ); 
	//waitKey(2000); 
    //cout<<"show finish\n";
	//system("pause");
	result.create(src.size(),src.type());
	result=src.clone();
    lapl.create( gray.size(), gray.type() );
	lapl2.create( gray.size(), gray.type() );
    mag.create( gray.size(), gray.type());
	gau.create( gray.size(), gray.type());
	dir.create( gray.size(), gray.type());
	lap_plus_gauss.create( gray.size(), gray.type());
	int rowN=gray.rows,colNum=gray.cols;
	for (int i = 1; i < rowN-1; i++)   
	        for (int j = 1; j < colNum-1; j++) 	
				   gau.at<uchar>(i,j)=Gaussian(i,j);


	for (int i = 1; i < rowN-1; i++)  
	{  
	        for (int j = 1; j < colNum-1; j++)        //Not cal the verge coordinate 
	        {  
	                        lapl.at<uchar>(i, j)=Laplacian(i,j);//Laplacian
							lapl2.at<uchar>(i, j)=Laplacian2(i,j);
							mag.at<uchar>(i, j)=Magnitude(i,j); //Magnitude
							lap_plus_gauss.at<uchar>(i,j)=Laplacian(i,j)/2+Magnitude(i,j);//Plus laplacian and gaussian
							cal_mag(i,j);       //calculate gradient at (i,j)
							cal_L(i,j);        //calculate direction vector
							cal_dp(i,j);       //calculate gradient.*direction vecter
	        }  
	}

	for(int i=1;i<rowN-1;i++)
		for(int j=1;j<colNum-1;j++)
			cal_fd(i,j);              //calculate fd(q,p)=acos(gradient.*direction vecter)
	for(int i=1;i<rowN-1;i++)
		for(int j=1;j<colNum-1;j++)
			cal_value(i,j);           //calculate I(q,p)=w1*(1-lap(q))+w2*fd(q,p)+w3*(1-fg(q))
    flag=0;
	cvNamedWindow("result",1); 
	cvSetMouseCallback( "result", on_mouse, 0 );
    IplImage iplimg;
    iplimg = IplImage(result);
	cvShowImage("result",&iplimg); 
	
	imshow("降噪",gau);
	imshow("拉普拉斯",lapl);
	imshow("拉普拉斯2",lapl2);
	imshow("梯度",mag);
	imshow("叠加",lap_plus_gauss);
	
	waitKey(0);   
  
    return 0;   
}  
uchar Laplacian(int x,int y)
{
	return (uchar)((double)(gau.at<uchar>(x,y-1)+gau.at<uchar>(x,y+1)+gau.at<uchar>(x-1,y)+gau.at<uchar>(x+1,y)-4*gau.at<uchar>(x,y))/2041*256);
}
uchar Laplacian2(int x,int y)
{
	return gau.at<uchar>(x,y-1)+gau.at<uchar>(x,y+1)+gau.at<uchar>(x-1,y)+gau.at<uchar>(x+1,y)-4*gau.at<uchar>(x,y);
}
uchar Gaussian(int x0,int y0)
{
	int sum=0;
	   for(int i=0;i<3;i++)
			for(int j=0;j<3;j++)
				sum+=gray.at<uchar>(x0+x[i],y0+y[j]);
	return (uchar)(sum/9);
}
uchar Magnitude(int x,int y)
{
	_mag[x][y]=sqrt((double)((gau.at<uchar>(x+1,y)-gau.at<uchar>(x,y))*(gau.at<uchar>(x+1,y)-gau.at<uchar>(x,y))+(gau.at<uchar>(x,y-1)-gau.at<uchar>(x,y))*(gau.at<uchar>(x,y-1)-gau.at<uchar>(x,y))));
	_mag[x][y]=_mag[x][y]/(361.625)*256;
	return (uchar)_mag[x][y];
}
double stmag(int x,int y)
{
	return sqrt((double)(magx[x][y]*magx[x][y]+magy[x][y]*magy[x][y]));
}
void cal_L(int i,int j)
{
	for(int k=0;k<8;k++){
		 if((k==0||k==2||k==5||k==7)&&directionx[k]*magy[i][j]-directiony[k]*magx[i][j]>=0)
			 gx[i][j][k]=directionx[k]*0.707,gy[i][j][k]=directiony[k]*0.707;
		 else if((k==0||k==2||k==5||k==7)&&directionx[k]*magy[i][j]-directiony[k]*magx[i][j]<0)
			 gx[i][j][k]=(-1)*directionx[k]*0.707,gy[i][j][k]=(-1)*directiony[k]*0.707;
		 else if((k==1||k==3||k==4||k==6)&&directionx[k]*magy[i][j]-directiony[k]*magx[i][j]>=0)
			 gx[i][j][k]=directionx[k],gy[i][j][k]=directiony[k];
		 else if((k==1||k==3||k==4||k==6)&&directionx[k]*magy[i][j]-directiony[k]*magx[i][j]<0)
			 gx[i][j][k]=-1*directionx[k],gy[i][j][k]=-1*directiony[k];
	}						
}			
void cal_dp(int i,int j)
{
	for(int k=0;k<8;k++)
		gx[i][j][k]=magy[i][j]*gx[i][j][k]-magx[i][j]*gy[i][j][k];
}
void cal_mag(int i,int j)
{
	                    double temp;
	                     //计算p的x,y分量
						magx[i][j]=gau.at<uchar>(i+1,j)-gau.at<uchar>(i,j);
						magy[i][j]=gau.at<uchar>(i,j+1)-gau.at<uchar>(i,j);
						//p分量单位化
						if((temp=stmag(i,j))!=0){
						magx[i][j]=magx[i][j]/temp;
						magy[i][j]=magy[i][j]/temp;
						}
}
void cal_fd(int i,int j)
{
	for(int k=0;k<8;k++)
		gx[i][j][k]=(double)((long long)(ex*((1/pi)*(acos((double)gx[i][j][k])+acos((double)gx[i+directionx[k]][j+directiony[k]][7-k])))))/ex;
}
void cal_value(int i,int j)
{
	for(int k=0;k<8;k++)
		gx[i][j][k]=((double)((long long)(ex*(Wz*(1-((double)lapl.at<uchar>(i, j)/256))+Wd*(gx[i][j][k])+Wg*(1-_mag[i][j]/256))))/ex);
}
void init()
{
	for(int i=0;i<maxn;i++)pre[i]=i,last[i]=inf,vis[i]=0;
}
void dij(int width,int height,int startx,int starty,int endx,int endy)
{
	NN tmp;
    heap q;
	//priority_queue<NN> q;
	cout<<"width ,  height  :"<<width<<"  "<<height<<endl;
	int v=0,tol=width*height,start=starty*width+startx,end=endy*width+endx,next;
	double minn;
	last[start]=0;
	tmp.v=start;tmp.t=0;
	q.build();
	q.push(tmp);
	while(q.size)
	{
		tmp=q.pop();;
		v=tmp.v;
		if(vis[v])continue;
		minn=tmp.t;
		vis[v]=1;
		for(int k=0;k<8;k++)
			if((v/width+directiony[k]<height&&v/width+directiony[k]>0)&&
				(v%width+directionx[k]<width&&v%width+directionx[k]>0)&&
				!vis[(next=v+directiony[k]*width+directionx[k])]&&minn+gx[v/width][v%width][k]<last[next])
			{
					last[next]=minn+gx[v/width][v%width][k];
					pre[next]=7-k;
					tmp.v=next;
					tmp.t=last[next];
					q.push(tmp);
			}
	}
}
IplImage* src0=0;    
IplImage* dst0=0;    
void draw(int y0,int x0)
{
	result.at<Vec3b>(y0,x0)[0]=255;
	result.at<Vec3b>(y0,x0)[1]=255;
	result.at<Vec3b>(y0,x0)[2]=0;
	for(int i=0;i<8;i++){
		if(y0+directiony[i]<0||y0+directiony[i]>=gray.size().height||
			x0+directionx[i]<0||x0+directionx[i]>=gray.size().width)continue;
		result.at<Vec3b>(y0+directiony[i],x0+directionx[i])[0]=255;
		result.at<Vec3b>(y0+directiony[i],x0+directionx[i])[1]=255;
		result.at<Vec3b>(y0+directiony[i],x0+directionx[i])[2]=0;
	}
}
void on_mouse( int event, int x, int y, int flags, void* ustc)    
{      
	if(x<=0||x>=gray.size().width||y<=0||y>=gray.size().height)return;
    cout<<"on__mouse\n";
    if( event == CV_EVENT_LBUTTONDOWN )    
    {    
		cout<<"mouse_click\n";
		flag=1;
		src=result.clone();
	    IplImage iplimg;
        iplimg = IplImage(src);
        CvPoint pt = cvPoint(x,y);    
        cvCircle( &iplimg, pt, 2,cvScalar(0,0,255,0) ,CV_FILLED, CV_AA, 0 );    
		startx=x;starty=y;
		init();
		dij(gray.size().width,gray.size().height,startx,starty,endx,endy);
        cvShowImage( "result", &iplimg );    
    }   
    else if(flag&&event == CV_EVENT_MOUSEMOVE )    
    {        
		cout<<"mouse_move\n";
		if(x==gray.size().width-1||y==gray.size().height-1)return;
		result=src.clone();
        CvPoint pt = cvPoint(x,y);    
		int end=y*gray.size().width+x,start=starty*gray.size().width+startx;
		long long clock=0;
	    while(end!=start)
	    {
			clock++;
			if(clock>1e6)return;
			draw(end/gray.size().width,end%gray.size().width);
		    end=end+directiony[pre[end]]*gray.size().width+directionx[pre[end]];
	    }
	    IplImage iplimg;
        iplimg = IplImage(result);
        cvShowImage( "result", &iplimg );
    }     
} 

