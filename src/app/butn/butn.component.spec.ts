import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ButnComponent } from './butn.component';

describe('ButnComponent', () => {
  let component: ButnComponent;
  let fixture: ComponentFixture<ButnComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ButnComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ButnComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
