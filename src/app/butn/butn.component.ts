import { Component, EventEmitter, OnInit, Output } from '@angular/core';

@Component({
  selector: 'app-butn',
  templateUrl: './butn.component.html',
  styleUrls: ['./butn.component.css']
})
export class ButnComponent implements OnInit {

  classes=['Sun', 'Flower', 'Umbrella', 'Pencil', 'Spoon', 'Tree', 'Mug', 'House', 'Bird', 'Hand'];

  constructor() { }

  ngOnInit(): void {
    this.classes.sort()
  }

  @Output() newItemEvent = new EventEmitter<string>();

  sendClassname(class_name:string){
    this.newItemEvent.emit(class_name);
  }

}
