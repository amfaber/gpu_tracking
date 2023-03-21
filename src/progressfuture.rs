use std::sync::{mpsc::{Receiver, Sender, self, TryRecvError, RecvError}, atomic::{AtomicBool, Ordering}, Arc, Mutex};
use std::thread::{Scope, ScopedJoinHandle};
use thiserror;


pub struct ScopedProgressFuture<'scope, R, P, J>
where
    R: Send + 'static, P: Send + Copy + Default + 'static, J: Send + 'scope,
{
    result_receiver: Receiver<R>,
    pub interrupter: Arc<AtomicBool>,
    pub progress: Arc<Mutex<P>>,
    job_sender: Sender<(J, Arc<AtomicBool>)>,
    pub queued_jobs: i32,
    pub handle: ScopedJoinHandle<'scope, ()>,
}

pub enum PollResult<R: Send + 'static, P: Send + 'static>{
    Done(R),
    Pending(P),
    NoJobRunning,
}

pub enum WaitResult<R: Send + 'static>{
    Done(R),
    NoJobRunning,
}


#[derive(Debug, thiserror::Error)]
pub enum Error{
    #[error("The thread as disconnected")]
    Disconnected,

    #[error("The mutex was poisoned")]
    Poisoned,

    #[error("Failed to send job")]
    SendError,
}

impl<'s, R, P, J> ScopedProgressFuture<'s, R, P, J>
where 
    R: Send + 'static, P: Send + Default + Copy + 'static, J: Send + 's
{
    pub fn from_interrupt_signal<'env: 's, F>(scope: &'s Scope<'s, 'env>, interrupter: Arc<AtomicBool>, function: F) -> Self
    where
        F: Send + 'static + Fn(J, &Arc<Mutex<P>>, &Arc<AtomicBool>) -> R
    {
        let progress = Arc::new(Mutex::new(P::default()));
        let (result_sender, result_receiver) = mpsc::channel();
        let (job_sender, job_receiver) = mpsc::channel();
        let handle = {
            let progress = progress.clone();
            scope.spawn(move||{
                loop{
                    match job_receiver.recv(){
                        Ok((job, interrupter)) => match result_sender.send(function(job, &progress, &interrupter)){
                            Ok(()) => (),
                            Err(_) => break,
                        },
                        Err(_) => break,
                    }
                }
            })
        };
        Self{
            result_receiver,
            interrupter,
            progress,
            job_sender,
            queued_jobs: 0,
            handle,
        }
    }

    pub fn new<'env: 's, F>(scope: &'s Scope<'s, 'env>, function: F) -> Self
    where
        F: Send + 'static + Fn(J, &Arc<Mutex<P>>, &Arc<AtomicBool>) -> R
    {
        let interrupter = Arc::new(AtomicBool::new(false));
        Self::from_interrupt_signal(scope, interrupter, function)
    }
    

    pub fn submit_new(&mut self, job: J) -> Result<(), Error>{
        self.interrupter = Arc::new(AtomicBool::new(false));
        *self.progress.lock().map_err(|_| Error::Poisoned)? = P::default();
        self.submit_same(job)
    }
    
    pub fn submit_same(&mut self, job: J) -> Result<(), Error>{
        let res = self.job_sender.send((job, self.interrupter.clone())).map_err(|_| Error::SendError);
        match res{
            Ok(_) => {
                self.queued_jobs += 1;
                return res
            },
            Err(_) => return res
        }
    }

    fn get_latest_result(&mut self) -> Result<Option<R>, Error>{
        let mut last = None;
        loop{
            match self.result_receiver.try_recv(){
                Ok(res) => {
                    self.queued_jobs -= 1;
                    last = Some(res)
                },
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return Err(Error::Disconnected)
            }
        }
        Ok(last)
    }
    
    pub fn poll(&mut self) -> Result<PollResult<R, P>, Error>{
        if self.queued_jobs == 0{
            return Ok(PollResult::NoJobRunning)
        }
        if let Some(res) = self.get_latest_result()?{
            return Ok(PollResult::Done(res))
        }
        
        match self.progress.lock(){
            Ok(prog) => Ok(PollResult::Pending(*prog)),
            Err(_) => Err(Error::Poisoned),
        }
    }

    pub fn read_progress(&self) -> Result<P, Error>{
        match self.progress.lock(){
            Ok(res) => Ok(*res),
            Err(_) => Err(Error::Poisoned),
        }
    }
    
    fn wait(&mut self) -> Result<WaitResult<R>, Error>{
        if self.queued_jobs == 0{
            return Ok(WaitResult::NoJobRunning)
        }
        match self.result_receiver.recv(){
            Ok(res) => {
                self.queued_jobs -= 1;
                return Ok(WaitResult::Done(res))
            },
            Err(RecvError) => return Err(Error::Disconnected)
        }
    }
    
    pub fn wait_for_latest(&mut self) -> Result<WaitResult<R>, Error>{
        if let Some(res) = self.get_latest_result()?{
            match self.queued_jobs{
                0 => return Ok(WaitResult::Done(res)),
                1 => {
                    return self.wait()
                }
                _ => panic!("I didn't manage the number of threads right")
            }
        }
        self.wait()
    }
        
    pub fn interrupt(&mut self){
        self.interrupter.store(true, Ordering::Relaxed);
    }
    
    pub fn interrupt_and_wait(&mut self) -> Result<WaitResult<R>, Error>{
        self.interrupter.store(true, Ordering::Relaxed);
        let res = self.wait_for_latest();
        self.interrupter.store(false, Ordering::Relaxed);
        res
    }

    pub fn interrupt_and_submit(&mut self, job: J) -> Result<(), Error>{
        self.interrupt();
        self.submit_new(job)
    }
    
    pub fn n_jobs(&self) -> i32{
        self.queued_jobs
    }
    
}

impl<'s, R, P, J> Drop for ScopedProgressFuture<'s, R, P, J>
where
    R: Send + 'static, P: Send + Default + Copy + 'static, J: Send + 's,
{
    fn drop(&mut self){
        self.interrupt();
    }
}








pub struct ProgressFuture<R, P, J>
where
    R: Send + 'static, P: Send + Copy + Default + 'static, J: Send + 'static,
{
    result_receiver: Receiver<R>,
    pub interrupter: Arc<AtomicBool>,
    pub progress: Arc<Mutex<P>>,
    job_sender: Sender<(J, Arc<AtomicBool>)>,
    pub queued_jobs: i32,
}

impl<R, P, J> ProgressFuture<R, P, J>
where 
    R: Send + 'static, P: Send + Default + Copy + 'static, J: Send + 'static
{
    pub fn from_interrupt_signal<F>(interrupter: Arc<AtomicBool>, function: F) -> Self
    where
        F: Send + 'static + Fn(J, &Arc<Mutex<P>>, &Arc<AtomicBool>) -> R
    {
        let progress = Arc::new(Mutex::new(P::default()));
        let (result_sender, result_receiver) = mpsc::channel();
        let (job_sender, job_receiver) = mpsc::channel();
        {
            let progress = progress.clone();
            std::thread::spawn(move||{
                loop{
                    match job_receiver.recv(){
                        Ok((job, interrupter)) => match result_sender.send(function(job, &progress, &interrupter)){
                            Ok(()) => (),
                            Err(_) => break,
                        },
                        Err(_) => break,
                    }
                }
            })
        };
        Self{
            result_receiver,
            interrupter,
            progress,
            job_sender,
            queued_jobs: 0,
        }
    }

    pub fn new<F>(function: F) -> Self
    where
        F: Send + 'static + Fn(J, &Arc<Mutex<P>>, &Arc<AtomicBool>) -> R
    {
        let interrupter = Arc::new(AtomicBool::new(false));
        Self::from_interrupt_signal(interrupter, function)
    }
    

    pub fn submit_new(&mut self, job: J) -> Result<(), Error>{
        self.interrupter = Arc::new(AtomicBool::new(false));
        *self.progress.lock().map_err(|_| Error::Poisoned)? = P::default();
        self.submit_same(job)
    }
    
    pub fn submit_same(&mut self, job: J) -> Result<(), Error>{
        let res = self.job_sender.send((job, self.interrupter.clone())).map_err(|_| Error::SendError);
        match res{
            Ok(_) => {
                self.queued_jobs += 1;
                return res
            },
            Err(_) => return res
        }
    }

    fn get_latest_result(&mut self) -> Result<Option<R>, Error>{
        let mut last = None;
        loop{
            match self.result_receiver.try_recv(){
                Ok(res) => {
                    self.queued_jobs -= 1;
                    last = Some(res)
                },
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return Err(Error::Disconnected)
            }
        }
        Ok(last)
    }
    
    pub fn poll(&mut self) -> Result<PollResult<R, P>, Error>{
        if self.queued_jobs == 0{
            return Ok(PollResult::NoJobRunning)
        }
        if let Some(res) = self.get_latest_result()?{
            return Ok(PollResult::Done(res))
        }
        
        match self.progress.lock(){
            Ok(prog) => Ok(PollResult::Pending(*prog)),
            Err(_) => Err(Error::Poisoned),
        }
    }
    
    fn wait(&mut self) -> Result<WaitResult<R>, Error>{
        if self.queued_jobs == 0{
            return Ok(WaitResult::NoJobRunning)
        }
        match self.result_receiver.recv(){
            Ok(res) => {
                self.queued_jobs -= 1;
                return Ok(WaitResult::Done(res))
            },
            Err(RecvError) => return Err(Error::Disconnected)
        }
    }
    
    pub fn wait_for_latest(&mut self) -> Result<WaitResult<R>, Error>{
        if let Some(res) = self.get_latest_result()?{
            match self.queued_jobs{
                0 => return Ok(WaitResult::Done(res)),
                1 => {
                    return self.wait()
                }
                _ => panic!("I didn't manage the number of threads right")
            }
        }
        self.wait()
    }
        
    pub fn interrupt(&mut self){
        self.interrupter.store(true, Ordering::Relaxed);
    }
    
    pub fn interrupt_and_wait(&mut self) -> Result<WaitResult<R>, Error>{
        self.interrupter.store(true, Ordering::Relaxed);
        let res = self.wait_for_latest();
        self.interrupter.store(false, Ordering::Relaxed);
        res
    }

    pub fn interrupt_and_submit(&mut self, job: J) -> Result<(), Error>{
        self.interrupt();
        self.submit_new(job)
    }
    
    pub fn n_jobs(&self) -> i32{
        self.queued_jobs
    }
    
}

impl<R, P, J> Drop for ProgressFuture<R, P, J>
where
    R: Send + 'static, P: Send + Default + Copy + 'static, J: Send,
{
    fn drop(&mut self){
        self.interrupt();
    }
}

