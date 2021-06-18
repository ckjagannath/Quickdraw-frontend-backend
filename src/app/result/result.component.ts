import { HttpClient } from '@angular/common/http';
import { Component, OnInit, AfterViewInit } from '@angular/core';
import { environment } from 'src/environments/environment';

@Component({
  selector: 'app-result',
  templateUrl: './result.component.html',
  styleUrls: ['./result.component.css']
})
export class ResultComponent implements OnInit,AfterViewInit {
  canvas;
  rect;
  context;
  isDrawing = false;
  prevX = 0;
  prevY = 0;
  newX;
  newY;
  body;
  class_name;
  result;
  constructor(private http:HttpClient) { 
  }
  ngOnInit(): void {
  }
  
  ngAfterViewInit(): void{
     
    this.canvas = document.getElementById('mycanvas');
    this.body = document.querySelector('.backgrnd');
    this.context = this.canvas.getContext('2d');
    
    // event.offsetX, event.offsetY gives the (x,y) offset from the edge of the canvas.
    
    // Add the event listeners for mousedown, mousemove, and mouseup
    this.canvas.addEventListener('mousedown', (e) => {
      this.rect=this.canvas.getBoundingClientRect();
      this.prevX = e.clientX-this.rect.left;
      this.prevY = e.clientY-this.rect.top;
      this.isDrawing = true;
    });
    this.canvas.addEventListener('touchstart', (e) => {
      this.rect=this.canvas.getBoundingClientRect();
      var touch = e.touches[0];
      this.prevX = touch.clientX-this.rect.left;
      this.prevY = touch.clientY-this.rect.top;
      this.isDrawing = true;
    });
    this.canvas.addEventListener('mousemove', (e) => {
      if (this.isDrawing === true) {
        this.rect=this.canvas.getBoundingClientRect();
        this.draw(this.context, this.prevX, this.prevY, this.newX=e.clientX-this.rect.left, this.newY=e.clientY-this.rect.top);
        this.prevX = this.newX;
        this.prevY = this.newY;
      }
    });
    this.canvas.addEventListener('touchmove', (e) => {
      var touch = e.touches[0];
      if (this.isDrawing === true) {
        this.rect=this.canvas.getBoundingClientRect();
        this.draw(this.context, this.prevX, this.prevY, this.newX=touch.clientX-this.rect.left, this.newY=touch.clientY-this.rect.top);
        this.prevX = this.newX;
        this.prevY = this.newY;
      }
    });
    
    this.canvas.addEventListener('mouseup', () => {
        this.isDrawing = false;
      }
    );
    this.canvas.addEventListener('touchend', (e) => {
      this.isDrawing = false;
    });
    this.canvas.addEventListener('mouseout', () => {
        this.isDrawing = false;
    }
    );
    window.addEventListener("resize",()=>{
      if (window.innerWidth<480){
        this.canvas.width=256;
        this.canvas.height=256;
      }
    })
    document.querySelector(".clear").addEventListener("click",()=>{
      this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    })
    this.body.addEventListener("touchstart",(e)=>{
      if (e.target==this.canvas){
        e.preventDefault();
      }
    })
    this.body.addEventListener("touchend",(e)=>{
      if (e.target==this.canvas){
        e.preventDefault();
      }
    })
    this.body.addEventListener("touchmove",(e)=>{
      if (e.target==this.canvas){
        e.preventDefault();
      }
    })
    }
    draw(ctx, x1, y1, x2, y2){
      this.context.beginPath();
      this.context.strokeStyle = 'black';
      this.context.lineWidth = 2;
      this.context.moveTo(x1, y1);
      this.context.lineTo(x2, y2);
      this.context.stroke();
      this.context.closePath();
    }
    save(){
      console.log('guess it selected');
      
      var date = Date.now();
      var filename = '_' + date + '.png';
      var image = this.canvas.toDataURL("image/png");
      this.http.post(
        environment.SERVER_URL +'/result',
        {filename, image},
        {responseType: 'text'}
      ).subscribe((res: any)=>{
        console.log('result= ',res);
        this.result ='It seems you have drawn ' + res; 
      })  
    }
}
