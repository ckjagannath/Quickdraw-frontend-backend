import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { CreateDatasetComponent } from './create-dataset/create-dataset.component';
import { HomeComponent } from './home/home.component';
import { ResultComponent } from './result/result.component';

const routes: Routes = [
  {
    path: 'create-dataset', component: CreateDatasetComponent
  },
  {
    path: 'result', component: ResultComponent
  },
  {
    path: 'home', component: HomeComponent
  },
  {
    path: '', redirectTo: 'home', pathMatch: 'full'
  }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
