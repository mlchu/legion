/* Copyright 2016 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "legion.h"
using namespace LegionRuntime::HighLevel;

/*
 * This example is a redux version of hello world 
 * which shows how launch a large array of tasks
 * using a single runtime call.  We also describe
 * the basic Legion types for arrays, domains,
 * and points and give examples of how they work.
 */

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  HELLO_WORLD_INDEX_ID,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  int num_points = 4;
  // See how many points to run
  const InputArgs &command_args = HighLevelRuntime::get_input_args();
  if (command_args.argc > 1)
  {
    num_points = atoi(command_args.argv[1]);
    assert(num_points > 0);
  }
  printf("Running hello world redux for %d points...\n", num_points);

  // To aid in describing structured data, Legion supports
  // a Rect type which is used to describe an array of
  // points.  Rects are templated on the number of 
  // dimensions that they describe.  To specify a Rect
  // a user gives two Points which specify the lower and
  // upper bounds on each dimension respectively.  Similar
  // to the Rect type, a Point type is templated on the
  // dimensions accessed.  Here we create a 1-D Rect which
  // we'll use to launch an array of tasks.  Note that the
  // bounds on Rects are inclusive.
  Rect<1> launch_bounds(Point<1>(0),Point<1>(num_points-1));
  // Rects can be converted to Domains.  Domains are useful
  // type which is equivalent to Rects but are not templated
  // on the number of dimensions.  Users can easily convert
  // between Domains and Rects using the 'from_rect' and
  // 'get_rect' methods.  Most Legion runtime calls will 
  // take Domains, but it often helps in application code
  // to have type checking support on the number of dimensions.
  Domain launch_domain = Domain::from_rect<1>(launch_bounds);

  // When we go to launch a large group of tasks in a single
  // call, we may want to pass different arguments to each
  // task.  ArgumentMaps allow the user to pass different
  // arguments to different points.  Note that ArgumentMaps
  // do not need to specify arguments for all points.  Legion
  // is intelligent about only passing arguments to the tasks
  // that have them.  Here we pass some values that we'll
  // use to illustrate how values get returned from an index
  // task launch.
  ArgumentMap arg_map;
  for (int i = 0; i < num_points; i++)
  {
    int input = i + 10;
    arg_map.set_point(DomainPoint::from_point<1>(Point<1>(i)),
        TaskArgument(&input,sizeof(input)));
  }
  // Legion supports launching an array of tasks with a 
  // single call.  We call these index tasks as we are launching
  // an array of tasks with one task for each point in the
  // array.  Index tasks are launched similar to single
  // tasks by using an index task launcher.  IndexLauncher
  // objects take the additional arguments of an ArgumentMap,
  // a TaskArgument which is a global argument that will
  // be passed to all tasks launched, and a domain describing
  // the points to be launched.
  IndexLauncher index_launcher(HELLO_WORLD_INDEX_ID,
                               launch_domain,
                               TaskArgument(NULL, 0),
                               arg_map);
  // Index tasks are launched the same as single tasks, but
  // return a future map which will store a future for all
  // points in the index space task launch.  Application
  // tasks can either wait on the future map for all tasks
  // in the index space to finish, or it can pull out 
  // individual futures for specific points on which to wait.
  FutureMap fm = runtime->execute_index_space(ctx, index_launcher);
  // Here we wait for all the futures to be ready
  fm.wait_all_results();
  // Now we can check that the future results that came back
  // from all the points in the index task are double 
  // their input.
  bool all_passed = true;
  for (int i = 0; i < num_points; i++)
  {
    int expected = 2*(i+10);
    int received = fm.get_result<int>(DomainPoint::from_point<1>(Point<1>(i)));
    if (expected != received)
    {
      printf("Check failed for point %d: %d != %d\n", i, expected, received);
      all_passed = false;
    }
  }
  if (all_passed)
    printf("All checks passed!\n");
}

int index_space_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, HighLevelRuntime *runtime)
{
  // The point for this task is available in the task
  // structure under the 'index_point' field.
  assert(task->index_point.get_dim() == 1); 
  printf("Hello world from task %d!\n", task->index_point.point_data[0]);
  // Values passed through an argument map are available 
  // through the local_args and local_arglen fields.
  assert(task->local_arglen == sizeof(int));
  int input = *((const int*)task->local_args);
  return (2*input);
}

int main(int argc, char **argv)
{
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/);
  HighLevelRuntime::register_legion_task<int, index_space_task>(HELLO_WORLD_INDEX_ID,
      Processor::LOC_PROC, false/*single*/, true/*index*/);

  return HighLevelRuntime::start(argc, argv);
}
