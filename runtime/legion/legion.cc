/* Copyright 2016 Stanford University, NVIDIA Corporation
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


#include "legion.h"
#include "runtime.h"
#include "legion_ops.h"
#include "legion_profiling.h"
#include "legion_allocation.h"

namespace LegionRuntime {
  namespace HighLevel {

    // If you add a logger, update the LEGION_EXTERN_LOGGER_DECLARATIONS
    // macro in legion_types.h
    Logger::Category log_run("runtime");
    Logger::Category log_task("tasks");
    Logger::Category log_index("index_spaces");
    Logger::Category log_field("field_spaces");
    Logger::Category log_region("regions");
    Logger::Category log_inst("instances");
    Logger::Category log_leak("leaks");
    Logger::Category log_variant("variants");
    Logger::Category log_allocation("allocation");
    Logger::Category log_prof("legion_prof");
    Logger::Category log_garbage("legion_gc");
    Logger::Category log_shutdown("shutdown");
    namespace LegionSpy {
      Logger::Category log_spy("legion_spy");
    };

    const LogicalRegion LogicalRegion::NO_REGION = LogicalRegion();
    const LogicalPartition LogicalPartition::NO_PART = LogicalPartition(); 

    /////////////////////////////////////////////////////////////
    // Mappable 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Mappable::Mappable(void)
      : map_id(0)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Task::Task(void)
      : Mappable(), args(NULL), arglen(0), local_args(NULL), local_arglen(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    unsigned Task::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return depth;
    }

    /////////////////////////////////////////////////////////////
    // Copy 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Copy::Copy(void)
      : Mappable(), parent_task(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    unsigned Copy::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_task->depth+1);
    }

    /////////////////////////////////////////////////////////////
    // Inline 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Inline::Inline(void)
      : Mappable(), parent_task(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    unsigned Inline::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_task->depth+1);
    }

    /////////////////////////////////////////////////////////////
    // Acquire 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Acquire::Acquire(void)
      : Mappable(), parent_task(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    unsigned Acquire::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_task->depth+1);
    }

    /////////////////////////////////////////////////////////////
    // Release 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Release::Release(void)
      : Mappable(), parent_task(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    unsigned Release::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_task->depth+1);
    }

    /////////////////////////////////////////////////////////////
    // IndexSpace 
    /////////////////////////////////////////////////////////////

    /*static*/ const IndexSpace IndexSpace::NO_SPACE = IndexSpace();

    //--------------------------------------------------------------------------
    IndexSpace::IndexSpace(IndexSpaceID _id, IndexTreeID _tid)
      : id(_id), tid(_tid)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexSpace::IndexSpace(void)
      : id(0), tid(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexSpace::IndexSpace(const IndexSpace &rhs)
      : id(rhs.id), tid(rhs.tid)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // IndexPartition 
    /////////////////////////////////////////////////////////////

    /*static*/ const IndexPartition IndexPartition::NO_PART = IndexPartition();

    //--------------------------------------------------------------------------
    IndexPartition::IndexPartition(IndexPartitionID _id, IndexTreeID _tid)
      : id(_id), tid(_tid)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexPartition::IndexPartition(void)
      : id(0), tid(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexPartition::IndexPartition(const IndexPartition &rhs)
      : id(rhs.id), tid(rhs.tid)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // FieldSpace 
    /////////////////////////////////////////////////////////////

    /*static*/ const FieldSpace FieldSpace::NO_SPACE = FieldSpace(0);

    //--------------------------------------------------------------------------
    FieldSpace::FieldSpace(unsigned _id)
      : id(_id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldSpace::FieldSpace(void)
      : id(0)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    FieldSpace::FieldSpace(const FieldSpace &rhs)
      : id(rhs.id)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Logical Region  
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalRegion::LogicalRegion(RegionTreeID tid, IndexSpace index, 
                                 FieldSpace field)
      : tree_id(tid), index_space(index), field_space(field)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalRegion::LogicalRegion(void)
      : tree_id(0), index_space(IndexSpace::NO_SPACE), 
        field_space(FieldSpace::NO_SPACE)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalRegion::LogicalRegion(const LogicalRegion &rhs)
      : tree_id(rhs.tree_id), index_space(rhs.index_space), 
        field_space(rhs.field_space)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Logical Partition 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalPartition::LogicalPartition(RegionTreeID tid, IndexPartition pid, 
                                       FieldSpace field)
      : tree_id(tid), index_partition(pid), field_space(field)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalPartition::LogicalPartition(void)
      : tree_id(0), index_partition(IndexPartition::NO_PART), 
        field_space(FieldSpace::NO_SPACE)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalPartition::LogicalPartition(const LogicalPartition &rhs)
      : tree_id(rhs.tree_id), index_partition(rhs.index_partition), 
        field_space(rhs.field_space)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Index Allocator 
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    IndexAllocator::IndexAllocator(void)
      : index_space(IndexSpace::NO_SPACE), allocator(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexAllocator::IndexAllocator(const IndexAllocator &rhs)
      : index_space(rhs.index_space), allocator(rhs.allocator)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexAllocator::IndexAllocator(IndexSpace space, IndexSpaceAllocator *a)
      : index_space(space), allocator(a) 
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexAllocator::~IndexAllocator(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexAllocator& IndexAllocator::operator=(const IndexAllocator &rhs)
    //--------------------------------------------------------------------------
    {
      index_space = rhs.index_space;
      allocator = rhs.allocator;
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Field Allocator 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldAllocator::FieldAllocator(void)
      : field_space(FieldSpace::NO_SPACE), parent(NULL), runtime(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldAllocator::FieldAllocator(const FieldAllocator &allocator)
      : field_space(allocator.field_space), parent(allocator.parent), 
        runtime(allocator.runtime)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldAllocator::FieldAllocator(FieldSpace f, Context p, 
                                   Runtime *rt)
      : field_space(f), parent(p), runtime(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldAllocator::~FieldAllocator(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldAllocator& FieldAllocator::operator=(const FieldAllocator &rhs)
    //--------------------------------------------------------------------------
    {
      field_space = rhs.field_space;
      parent = rhs.parent;
      runtime = rhs.runtime;
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Argument Map 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ArgumentMap::ArgumentMap(void)
    //--------------------------------------------------------------------------
    {
      impl = legion_new<ArgumentMap::Impl>();
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->add_reference();
    }

    //--------------------------------------------------------------------------
    ArgumentMap::ArgumentMap(const ArgumentMap &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    ArgumentMap::ArgumentMap(Impl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    ArgumentMap::~ArgumentMap(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        // Remove our reference and if we were the
        // last reference holder, then delete it
        if (impl->remove_reference())
        {
          legion_delete(impl);
        }
        impl = NULL;
      }
    }
    
    //--------------------------------------------------------------------------
    ArgumentMap& ArgumentMap::operator=(const ArgumentMap &rhs)
    //--------------------------------------------------------------------------
    {
      // Check to see if our current impl is not NULL,
      // if so remove our reference
      if (impl != NULL)
      {
        if (impl->remove_reference())
        {
          legion_delete(impl);
        }
      }
      impl = rhs.impl;
      // Add our reference to the new impl
      if (impl != NULL)
      {
        impl->add_reference();
      }
      return *this;
    }

    //--------------------------------------------------------------------------
    bool ArgumentMap::has_point(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      return impl->has_point(point);
    }

    //--------------------------------------------------------------------------
    void ArgumentMap::set_point(const DomainPoint &point, 
                                const TaskArgument &arg, bool replace/*= true*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->set_point(point, arg, replace);
    }

    //--------------------------------------------------------------------------
    bool ArgumentMap::remove_point(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      return impl->remove_point(point);
    }

    //--------------------------------------------------------------------------
    TaskArgument ArgumentMap::get_point(const DomainPoint &point) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      return impl->get_point(point);
    }

    /////////////////////////////////////////////////////////////
    // Predicate 
    /////////////////////////////////////////////////////////////

    const Predicate Predicate::TRUE_PRED = Predicate(true);
    const Predicate Predicate::FALSE_PRED = Predicate(false);

    //--------------------------------------------------------------------------
    Predicate::Predicate(void)
      : impl(NULL), const_value(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Predicate::Predicate(const Predicate &p)
    //--------------------------------------------------------------------------
    {
      const_value = p.const_value;
      impl = p.impl;
      if (impl != NULL)
        impl->add_predicate_reference();
    }

    //--------------------------------------------------------------------------
    Predicate::Predicate(bool value)
      : impl(NULL), const_value(value)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Predicate::Predicate(Predicate::Impl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_predicate_reference();
    }

    //--------------------------------------------------------------------------
    Predicate::~Predicate(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        impl->remove_predicate_reference();
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    Predicate& Predicate::operator=(const Predicate &rhs)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->remove_predicate_reference();
      const_value = rhs.const_value;
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_predicate_reference();
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Lock 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Lock::Lock(void)
      : reservation_lock(Reservation::NO_RESERVATION)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Lock::Lock(Reservation r)
      : reservation_lock(r)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool Lock::operator<(const Lock &rhs) const
    //--------------------------------------------------------------------------
    {
      return (reservation_lock < rhs.reservation_lock);
    }

    //--------------------------------------------------------------------------
    bool Lock::operator==(const Lock &rhs) const
    //--------------------------------------------------------------------------
    {
      return (reservation_lock == rhs.reservation_lock);
    }

    //--------------------------------------------------------------------------
    void Lock::acquire(unsigned mode /*=0*/, bool exclusive /*=true*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(reservation_lock.exists());
#endif
      Event lock_event = reservation_lock.acquire(mode,exclusive);
      if (!lock_event.has_triggered())
      {
        Processor proc = Processor::get_executing_processor();
        Internal *rt = Internal::get_runtime(proc);
        rt->pre_wait(proc);
        lock_event.wait();
        rt->post_wait(proc);
      }
    }

    //--------------------------------------------------------------------------
    void Lock::release(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(reservation_lock.exists());
#endif
      reservation_lock.release();
    }

    /////////////////////////////////////////////////////////////
    // Grant 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Grant::Grant(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Grant::Grant(Grant::Impl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    Grant::Grant(const Grant &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    Grant::~Grant(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          legion_delete(impl);
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    Grant& Grant::operator=(const Grant &rhs)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          legion_delete(impl);
      }
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_reference();
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Phase Barrier 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhaseBarrier::PhaseBarrier(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhaseBarrier::PhaseBarrier(Barrier b)
      : phase_barrier(b)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool PhaseBarrier::operator<(const PhaseBarrier &rhs) const
    //--------------------------------------------------------------------------
    {
      return (phase_barrier < rhs.phase_barrier);
    }

    //--------------------------------------------------------------------------
    bool PhaseBarrier::operator==(const PhaseBarrier &rhs) const
    //--------------------------------------------------------------------------
    {
      return (phase_barrier == rhs.phase_barrier);
    }

    //--------------------------------------------------------------------------
    void PhaseBarrier::arrive(unsigned count /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(phase_barrier.exists());
#endif
      phase_barrier.arrive(count);
    }

    //--------------------------------------------------------------------------
    void PhaseBarrier::wait(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(phase_barrier.exists());
#endif
      if (!phase_barrier.has_triggered())
      {
        Processor proc = Processor::get_executing_processor();
        Internal *rt = Internal::get_runtime(proc);
        rt->pre_wait(proc);
        phase_barrier.wait();
        rt->post_wait(proc);
      }
    }

    //--------------------------------------------------------------------------
    void PhaseBarrier::alter_arrival_count(int delta)
    //--------------------------------------------------------------------------
    {
      phase_barrier.alter_arrival_count(delta);
    }

    /////////////////////////////////////////////////////////////
    // Dynamic Collective 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DynamicCollective::DynamicCollective(void)
      : PhaseBarrier(), redop(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DynamicCollective::DynamicCollective(Barrier b, ReductionOpID r)
      : PhaseBarrier(b), redop(r)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void DynamicCollective::arrive(const void *value, size_t size, 
                                   unsigned count /*=1*/)
    //--------------------------------------------------------------------------
    {
      phase_barrier.arrive(count, Event::NO_EVENT, value, size); 
    }

    /////////////////////////////////////////////////////////////
    // Region Requirement 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, 
                                        const std::set<FieldID> &priv_fields,
                                        const std::vector<FieldID> &inst_fields,
                                        PrivilegeMode _priv, 
                                        CoherenceProperty _prop, 
                                        LogicalRegion _parent,
				        MappingTagID _tag, bool _verified)
      : region(_handle), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(SINGULAR), projection(0)
    //--------------------------------------------------------------------------
    { 
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this)) // Shouldn't use this constructor for reductions
      {
        log_region.error("ERROR: Use different RegionRequirement "
                                "constructor for reductions");
        assert(false);
        exit(ERROR_USE_REDUCTION_REGION_REQ);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalPartition pid, 
                ProjectionID _proj, 
                const std::set<FieldID> &priv_fields,
                const std::vector<FieldID> &inst_fields,
                PrivilegeMode _priv, CoherenceProperty _prop,
                LogicalRegion _parent, MappingTagID _tag, bool _verified)
      : partition(pid), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG),
        handle_type(PART_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    { 
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this))
      {
        log_region.error("ERROR: Use different RegionRequirement "
                                "constructor for reductions");
        assert(false);
        exit(ERROR_USE_REDUCTION_REGION_REQ);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, 
                ProjectionID _proj,
                const std::set<FieldID> &priv_fields,
                const std::vector<FieldID> &inst_fields,
                PrivilegeMode _priv, CoherenceProperty _prop,
                LogicalRegion _parent, MappingTagID _tag, bool _verified)
      : region(_handle), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG),
        handle_type(REG_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    {
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this))
      {
        log_region.error("ERROR: Use different RegionRequirement "
                                "constructor for reductions");
        assert(false);
        exit(ERROR_USE_REDUCTION_REGION_REQ);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle,  
                                    const std::set<FieldID> &priv_fields,
                                    const std::vector<FieldID> &inst_fields,
                                    ReductionOpID op, CoherenceProperty _prop, 
                                    LogicalRegion _parent, MappingTagID _tag, 
                                    bool _verified)
      : region(_handle), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(SINGULAR)
    //--------------------------------------------------------------------------
    {
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
#ifdef DEBUG_HIGH_LEVEL
      if (redop == 0)
      {
        log_region.error("Zero is not a valid ReductionOpID");
        assert(false);
        exit(ERROR_RESERVED_REDOP_ID);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalPartition pid, 
                        ProjectionID _proj,  
                        const std::set<FieldID> &priv_fields,
                        const std::vector<FieldID> &inst_fields,
                        ReductionOpID op, CoherenceProperty _prop,
                        LogicalRegion _parent, MappingTagID _tag, 
                        bool _verified)
      : partition(pid), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG),
        handle_type(PART_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    {
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
#ifdef DEBUG_HIGH_LEVEL
      if (redop == 0)
      {
        log_region.error("Zero is not a valid ReductionOpID");
        assert(false);
        exit(ERROR_RESERVED_REDOP_ID);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, 
                        ProjectionID _proj,
                        const std::set<FieldID> &priv_fields,
                        const std::vector<FieldID> &inst_fields,
                        ReductionOpID op, CoherenceProperty _prop,
                        LogicalRegion _parent, MappingTagID _tag, 
                        bool _verified)
      : region(_handle), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG),
        handle_type(REG_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    {
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
#ifdef DEBUG_HIGH_LEVEL
      if (redop == 0)
      {
        log_region.error("Zero is not a valid ReductionOpID");
        assert(false);
        exit(ERROR_RESERVED_REDOP_ID);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, 
                                         PrivilegeMode _priv, 
                                         CoherenceProperty _prop, 
                                         LogicalRegion _parent,
					 MappingTagID _tag, 
                                         bool _verified)
      : region(_handle), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(SINGULAR)
    //--------------------------------------------------------------------------
    { 
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this)) // Shouldn't use this constructor for reductions
      {
        log_region.error("ERROR: Use different RegionRequirement "
                                "constructor for reductions");
        assert(false);
        exit(ERROR_USE_REDUCTION_REGION_REQ);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalPartition pid, 
                                         ProjectionID _proj, 
                                         PrivilegeMode _priv, 
                                         CoherenceProperty _prop,
                                         LogicalRegion _parent, 
                                         MappingTagID _tag, 
                                         bool _verified)
      : partition(pid), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(PART_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    { 
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this))
      {
        log_region.error("ERROR: Use different RegionRequirement "
                                "constructor for reductions");
        assert(false);
        exit(ERROR_USE_REDUCTION_REGION_REQ);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, 
                                         ProjectionID _proj,
                                         PrivilegeMode _priv, 
                                         CoherenceProperty _prop,
                                         LogicalRegion _parent, 
                                         MappingTagID _tag, 
                                         bool _verified)
      : region(_handle), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(REG_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this))
      {
        log_region.error("ERROR: Use different RegionRequirement "
                                "constructor for reductions");
        assert(false);
        exit(ERROR_USE_REDUCTION_REGION_REQ);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle,  
                                         ReductionOpID op, 
                                         CoherenceProperty _prop, 
                                         LogicalRegion _parent, 
                                         MappingTagID _tag, 
                                         bool _verified)
      : region(_handle), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(SINGULAR)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (redop == 0)
      {
        log_region.error("Zero is not a valid ReductionOpID");
        assert(false);
        exit(ERROR_RESERVED_REDOP_ID);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalPartition pid, 
                                         ProjectionID _proj,  
                                         ReductionOpID op, 
                                         CoherenceProperty _prop,
                                         LogicalRegion _parent, 
                                         MappingTagID _tag, 
                                         bool _verified)
      : partition(pid), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(PART_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (redop == 0)
      {
        log_region.error("Zero is not a valid ReductionOpID");
        assert(false);
        exit(ERROR_RESERVED_REDOP_ID);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, 
                                         ProjectionID _proj,
                                         ReductionOpID op, 
                                         CoherenceProperty _prop,
                                         LogicalRegion _parent, 
                                         MappingTagID _tag, 
                                         bool _verified)
      : region(_handle), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(REG_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (redop == 0)
      {
        log_region.error("Zero is not a valid ReductionOpID");
        assert(false);
        exit(ERROR_RESERVED_REDOP_ID);
      }
#endif
    }

    //--------------------------------------------------------------------------
    bool RegionRequirement::operator==(const RegionRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      if ((handle_type == rhs.handle_type) && (privilege == rhs.privilege) &&
          (prop == rhs.prop) && (parent == rhs.parent) && (redop == rhs.redop)
          && (tag == rhs.tag) && (flags == rhs.flags))
      {
        if (((handle_type == SINGULAR) && (region == rhs.region)) ||
            ((handle_type == PART_PROJECTION) && (partition == rhs.partition) && 
             (projection == rhs.projection)) ||
            ((handle_type == REG_PROJECTION) && (region == rhs.region)))
        {
          if ((privilege_fields.size() == rhs.privilege_fields.size()) &&
              (instance_fields.size() == rhs.instance_fields.size()))
          {
            return ((privilege_fields == rhs.privilege_fields) 
                && (instance_fields == rhs.instance_fields));
          }
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool RegionRequirement::operator<(const RegionRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      if (handle_type < rhs.handle_type)
        return true;
      else if (handle_type > rhs.handle_type)
        return false;
      else
      {
        if (privilege < rhs.privilege)
          return true;
        else if (privilege > rhs.privilege)
          return false;
        else
        {
          if (prop < rhs.prop)
            return true;
          else if (prop > rhs.prop)
            return false;
          else
          {
            if (parent < rhs.parent)
              return true;
            else if (!(parent == rhs.parent)) // therefore greater than
              return false;
            else
            {
              if (redop < rhs.redop)
                return true;
              else if (redop > rhs.redop)
                return false;
              else
              {
                if (tag < rhs.tag)
                  return true;
                else if (tag > rhs.tag)
                  return false;
                else
                {
                  if (flags < rhs.flags)
                    return true;
                  else if (flags > rhs.flags)
                    return false;
                  else
                  {
                    if (privilege_fields < rhs.privilege_fields)
                      return true;
                    else if (privilege_fields > rhs.privilege_fields)
                      return false;
                    else
                    {
                      if (instance_fields < rhs.instance_fields)
                        return true;
                      else if (instance_fields > rhs.instance_fields)
                        return false;
                      else
                      {
                        if (handle_type == SINGULAR)
                          return (region < rhs.region);
                        else if (handle_type == PART_PROJECTION)
                        {
                          if (partition < rhs.partition)
                            return true;
                          // therefore greater than
                          else if (partition != rhs.partition) 
                            return false;
                          else
                            return (projection < rhs.projection);
                        }
                        else
                        {
                          if (region < rhs.region)
                            return true;
                          else if (region != rhs.region)
                            return false;
                          else
                            return (projection < rhs.projection);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

#ifdef PRIVILEGE_CHECKS
    //--------------------------------------------------------------------------
    AccessorPrivilege RegionRequirement::get_accessor_privilege(void) const
    //--------------------------------------------------------------------------
    {
      switch (privilege)
      {
        case NO_ACCESS:
          return ACCESSOR_NONE;
        case READ_ONLY:
          return ACCESSOR_READ;
        case READ_WRITE:
        case WRITE_DISCARD:
          return ACCESSOR_ALL;
        case REDUCE:
          return ACCESSOR_REDUCE;
        default:
          assert(false);
      }
      return ACCESSOR_NONE;
    }
#endif

    //--------------------------------------------------------------------------
    bool RegionRequirement::has_field_privilege(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      return (privilege_fields.find(fid) != privilege_fields.end());
    }

    //--------------------------------------------------------------------------
    void RegionRequirement::copy_without_mapping_info(
                                                  const RegionRequirement &rhs)
    //--------------------------------------------------------------------------
    {
      region = rhs.region;
      partition = rhs.partition;
      privilege_fields = rhs.privilege_fields;
      instance_fields = rhs.instance_fields;
      privilege = rhs.privilege;
      prop = rhs.prop;
      parent = rhs.parent;
      redop = rhs.redop;
      tag = rhs.tag;
      flags = rhs.flags;
      handle_type = rhs.handle_type;
      projection = rhs.projection;
    }

    //--------------------------------------------------------------------------
    void RegionRequirement::initialize_mapping_fields(void)
    //--------------------------------------------------------------------------
    {
      premapped = false;
      must_early_map = false;
      restricted = false;
      max_blocking_factor = 1;
      current_instances.clear();
      virtual_map = false;
      early_map = false;
      enable_WAR_optimization = false;
      reduction_list = false;
      make_persistent = false;
      blocking_factor = 1;
      target_ranking.clear();
      additional_fields.clear();
      mapping_failed = false;
      selected_memory = Memory::NO_MEMORY;
    }

    /////////////////////////////////////////////////////////////
    // Index Space Requirement 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexSpaceRequirement::IndexSpaceRequirement(void)
      : handle(IndexSpace::NO_SPACE), privilege(NO_MEMORY), 
        parent(IndexSpace::NO_SPACE), verified(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexSpaceRequirement::IndexSpaceRequirement(IndexSpace _handle, 
                                                 AllocateMode _priv,
                                                 IndexSpace _parent, 
                                                 bool _verified /*=false*/)
      : handle(_handle), privilege(_priv), parent(_parent), verified(_verified)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceRequirement::operator<(
                                        const IndexSpaceRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      if (handle < rhs.handle)
        return true;
      else if (handle != rhs.handle) // therefore greater than
        return false;
      else
      {
        if (privilege < rhs.privilege)
          return true;
        else if (privilege > rhs.privilege)
          return false;
        else
        {
          if (parent < rhs.parent)
            return true;
          else if (parent != rhs.parent) // therefore greater than
            return false;
          else
            return verified < rhs.verified;
        }
      }
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceRequirement::operator==(
                                        const IndexSpaceRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      return (handle == rhs.handle) && (privilege == rhs.privilege) &&
             (parent == rhs.parent) && (verified == rhs.verified);
    }

    /////////////////////////////////////////////////////////////
    // Field Space Requirement 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldSpaceRequirement::FieldSpaceRequirement(void)
      : handle(FieldSpace::NO_SPACE), privilege(NO_MEMORY), verified(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldSpaceRequirement::FieldSpaceRequirement(FieldSpace _handle, 
                                                 AllocateMode _priv,
                                                 bool _verified /*=false*/)
      : handle(_handle), privilege(_priv), verified(_verified)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceRequirement::operator<(
                                        const FieldSpaceRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      if (handle < rhs.handle)
        return true;
      else if (!(handle == rhs.handle)) // therefore greater than
        return false;
      else
      {
        if (privilege < rhs.privilege)
          return true;
        else if (privilege > rhs.privilege)
          return false;
        else
          return verified < rhs.verified;
      }
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceRequirement::operator==(
                                        const FieldSpaceRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      return (handle == rhs.handle) && 
              (privilege == rhs.privilege) && (verified == rhs.verified);
    }

    /////////////////////////////////////////////////////////////
    // TaskLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TaskLauncher::TaskLauncher(void)
      : task_id(0), argument(TaskArgument()), predicate(Predicate::TRUE_PRED),
        map_id(0), tag(0), point(DomainPoint())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TaskLauncher::TaskLauncher(Processor::TaskFuncID tid, TaskArgument arg,
                               Predicate pred /*= Predicate::TRUE_PRED*/,
                               MapperID mid /*=0*/, MappingTagID t /*=0*/)
      : task_id(tid), argument(arg), predicate(pred), 
        map_id(mid), tag(t), point(DomainPoint())
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // IndexLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexLauncher::IndexLauncher(void)
      : task_id(0), launch_domain(Domain::NO_DOMAIN), 
        global_arg(TaskArgument()), argument_map(ArgumentMap()), 
        predicate(Predicate::TRUE_PRED), must_parallelism(false), 
        map_id(0), tag(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexLauncher::IndexLauncher(Processor::TaskFuncID tid, Domain dom,
                                 TaskArgument global,
                                 ArgumentMap map,
                                 Predicate pred /*= Predicate::TRUE_PRED*/,
                                 bool must /*=false*/, MapperID mid /*=0*/,
                                 MappingTagID t /*=0*/)
      : task_id(tid), launch_domain(dom), global_arg(global), 
        argument_map(map), predicate(pred), must_parallelism(must),
        map_id(mid), tag(t)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // InlineLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InlineLauncher::InlineLauncher(void)
      : map_id(0), tag(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InlineLauncher::InlineLauncher(const RegionRequirement &req,
                                   MapperID mid /*=0*/, MappingTagID t /*=0*/)
      : requirement(req), map_id(mid), tag(t)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // CopyLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CopyLauncher::CopyLauncher(Predicate pred /*= Predicate::TRUE_PRED*/,
                               MapperID mid /*=0*/, MappingTagID t /*=0*/)
      : predicate(pred), map_id(mid), tag(t)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // AcquireLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AcquireLauncher::AcquireLauncher(LogicalRegion reg, LogicalRegion par,
                                     PhysicalRegion phy,
                                     Predicate pred /*= Predicate::TRUE_PRED*/,
                                     MapperID id /*=0*/, MappingTagID t /*=0*/)
      : logical_region(reg), parent_region(par), physical_region(phy), 
        predicate(pred), map_id(id), tag(t)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // ReleaseLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReleaseLauncher::ReleaseLauncher(LogicalRegion reg, LogicalRegion par,
                                     PhysicalRegion phy,
                                     Predicate pred /*= Predicate::TRUE_PRED*/,
                                     MapperID id /*=0*/, MappingTagID t /*=0*/)
      : logical_region(reg), parent_region(par), physical_region(phy), 
        predicate(pred), map_id(id), tag(t)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // MustEpochLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MustEpochLauncher::MustEpochLauncher(MapperID id /*= 0*/,   
                                         MappingTagID tag/*= 0*/)
      : map_id(id), mapping_tag(tag)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // LayoutConstraintRegistrar
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LayoutConstraintRegistrar::LayoutConstraintRegistrar(void)
      : handle(FieldSpace::NO_SPACE), layout_name(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LayoutConstraintRegistrar::LayoutConstraintRegistrar(FieldSpace h,
                                                  const char *layout/*= NULL*/)
      : handle(h), layout_name(layout)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // TaskVariantRegistrar 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TaskVariantRegistrar::TaskVariantRegistrar(void)
      : task_id(0), global_registration(true), task_variant_name(NULL), 
        leaf_variant(false), inner_variant(false), idempotent_variant(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TaskVariantRegistrar::TaskVariantRegistrar(TaskID tid, bool global/*=true*/, 
                                               const char *name/*= NULL*/)
      : task_id(tid), global_registration(global), task_variant_name(name), 
        leaf_variant(false), inner_variant(false), idempotent_variant(false)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // MPILegionHandshake 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MPILegionHandshake::MPILegionHandshake(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MPILegionHandshake::MPILegionHandshake(const MPILegionHandshake &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    MPILegionHandshake::~MPILegionHandshake(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          legion_delete(impl);
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    MPILegionHandshake::MPILegionHandshake(MPILegionHandshake::Impl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    MPILegionHandshake& MPILegionHandshake::operator=(
                                                  const MPILegionHandshake &rhs)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          legion_delete(impl);
      }
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    void MPILegionHandshake::mpi_handoff_to_legion(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->mpi_handoff_to_legion();
    }

    //--------------------------------------------------------------------------
    void MPILegionHandshake::mpi_wait_on_legion(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->mpi_wait_on_legion();
    }

    //--------------------------------------------------------------------------
    void MPILegionHandshake::legion_handoff_to_mpi(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->legion_handoff_to_mpi();
    }

    //--------------------------------------------------------------------------
    void MPILegionHandshake::legion_wait_on_mpi(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->legion_wait_on_mpi();
    }

    /////////////////////////////////////////////////////////////
    // Future 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Future::Future(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Future::Future(const Future &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_base_gc_ref(FUTURE_HANDLE_REF);
    }

    //--------------------------------------------------------------------------
    Future::~Future(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_base_gc_ref(FUTURE_HANDLE_REF))
          legion_delete(impl);
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    Future::Future(Future::Impl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_base_gc_ref(FUTURE_HANDLE_REF);
    }

    //--------------------------------------------------------------------------
    Future& Future::operator=(const Future &rhs)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_base_gc_ref(FUTURE_HANDLE_REF))
          legion_delete(impl);
      }
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_base_gc_ref(FUTURE_HANDLE_REF);
      return *this;
    }

    //--------------------------------------------------------------------------
    void Future::get_void_result(void) const
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->get_void_result();
    }

    //--------------------------------------------------------------------------
    bool Future::is_empty(bool block /*= true*/) const
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        return impl->is_empty(block);
      return true;
    }

    //--------------------------------------------------------------------------
    void* Future::get_untyped_result(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        return impl->get_untyped_result();
      else
        return NULL;
    }

    /////////////////////////////////////////////////////////////
    // Future Map 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureMap::FutureMap(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureMap::FutureMap(const FutureMap &map)
      : impl(map.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    FutureMap::FutureMap(FutureMap::Impl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    FutureMap::~FutureMap(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          legion_delete(impl);
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    FutureMap& FutureMap::operator=(const FutureMap &rhs)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          legion_delete(impl);
      }
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    Future FutureMap::get_future(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      return impl->get_future(point);
    }

    //--------------------------------------------------------------------------
    void FutureMap::get_void_result(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->get_void_result(point);
    }

    //--------------------------------------------------------------------------
    void FutureMap::wait_all_results(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->wait_all_results();
    }

    /////////////////////////////////////////////////////////////
    // Physical Region 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalRegion::PhysicalRegion(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalRegion::PhysicalRegion(const PhysicalRegion &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    PhysicalRegion::PhysicalRegion(PhysicalRegion::Impl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    PhysicalRegion::~PhysicalRegion(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          legion_delete(impl);
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    PhysicalRegion& PhysicalRegion::operator=(const PhysicalRegion &rhs)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          legion_delete(impl);
      }
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    void PhysicalRegion::wait_until_valid(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->wait_until_valid();
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegion::is_valid(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      return impl->is_valid();
    }

    //--------------------------------------------------------------------------
    LogicalRegion PhysicalRegion::get_logical_region(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      return impl->get_logical_region();
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic>
      PhysicalRegion::get_accessor(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      return impl->get_accessor();
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic>
      PhysicalRegion::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      return impl->get_field_accessor(fid);
    }

    //--------------------------------------------------------------------------
    void PhysicalRegion::get_memories(std::set<Memory>& memories) const
    //--------------------------------------------------------------------------
    {
      impl->get_memories(memories);
    }

    //--------------------------------------------------------------------------
    void PhysicalRegion::get_fields(std::vector<FieldID>& fields) const
    //--------------------------------------------------------------------------
    {
      impl->get_fields(fields);
    }

    /////////////////////////////////////////////////////////////
    // Index Iterator  
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexIterator::IndexIterator(const Domain &dom)
      : enumerator(dom.get_index_space().get_valid_mask().enumerate_enabled())
    //--------------------------------------------------------------------------
    {
      finished = !(enumerator->get_next(current_pointer,remaining_elmts));
    }

    //--------------------------------------------------------------------------
    IndexIterator::IndexIterator(Runtime *rt, Context ctx,
                                 IndexSpace space)
    //--------------------------------------------------------------------------
    {
      Domain dom = rt->get_index_space_domain(ctx, space);
      enumerator = dom.get_index_space().get_valid_mask().enumerate_enabled();
      finished = !(enumerator->get_next(current_pointer,remaining_elmts));
    }

    //--------------------------------------------------------------------------
    IndexIterator::IndexIterator(Runtime *rt, Context ctx,
                                 LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      Domain dom = rt->get_index_space_domain(ctx, handle.get_index_space());
      enumerator = dom.get_index_space().get_valid_mask().enumerate_enabled();
      finished = !(enumerator->get_next(current_pointer,remaining_elmts));
    }

    //--------------------------------------------------------------------------
    IndexIterator::IndexIterator(const IndexIterator &rhs)
      : enumerator(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    IndexIterator::~IndexIterator(void)
    //--------------------------------------------------------------------------
    {
      delete enumerator;
    }

    //--------------------------------------------------------------------------
    IndexIterator& IndexIterator::operator=(const IndexIterator &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Task Variant Collection
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    bool TaskVariantCollection::has_variant(Processor::Kind kind, 
                                            bool single,
                                            bool index_space)
    //--------------------------------------------------------------------------
    {
      for (std::map<VariantID,Variant>::const_iterator it = variants.begin();
            it != variants.end(); it++)
      {
        if ((it->second.proc_kind == kind) && 
            ((it->second.single_task <= single) || 
            (it->second.index_space <= index_space)))
        {
          return true;
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    VariantID TaskVariantCollection::get_variant(Processor::Kind kind, 
                                                 bool single,
                                                 bool index_space)
    //--------------------------------------------------------------------------
    {
      for (std::map<VariantID,Variant>::const_iterator it = variants.begin();
            it != variants.end(); it++)
      {
        if ((it->second.proc_kind == kind) && 
            ((it->second.single_task <= single) || 
            (it->second.index_space <= index_space)))
        {
          return it->first;
        }
      }
      log_variant.error("User task %s (ID %d) has no registered variants "
                              "for processors of kind %d and index space %d",
                              name, user_id, kind, index_space);
#ifdef DEBUG_HIGH_LEVEL
      assert(false);
#endif
      exit(ERROR_UNREGISTERED_VARIANT);
      return 0;
    }

    //--------------------------------------------------------------------------
    bool TaskVariantCollection::has_variant(VariantID vid)
    //--------------------------------------------------------------------------
    {
      return (variants.find(vid) != variants.end());
    }

    //--------------------------------------------------------------------------
    const TaskVariantCollection::Variant& TaskVariantCollection::get_variant(
                                                                  VariantID vid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(variants.find(vid) != variants.end());
#endif
      return variants[vid];
    }

    //--------------------------------------------------------------------------
    void TaskVariantCollection::add_variant(Processor::TaskFuncID low_id, 
                                            Processor::Kind kind, 
                                            bool single, bool index,
                                            bool inner, bool leaf,
                                            VariantID vid)
    //--------------------------------------------------------------------------
    {
      if (vid == AUTO_GENERATE_ID)
      {
        for (unsigned idx = 0; idx < AUTO_GENERATE_ID; idx++)
        {
          if (variants.find(idx) == variants.end())
          {
            vid = idx;
            break;
          }
        }
      }
      variants[vid] = Variant(low_id, kind, single, index, inner, leaf, vid);
    }

    //--------------------------------------------------------------------------
    const TaskVariantCollection::Variant& TaskVariantCollection::select_variant(
                                  bool single, bool index, Processor::Kind kind)
    //--------------------------------------------------------------------------
    {
      for (std::map<VariantID,Variant>::const_iterator it = variants.begin();
            it != variants.end(); it++)
      {
        if ((it->second.proc_kind == kind) && 
            (it->second.single_task <= single) &&
            (it->second.index_space <= index))
        {
          return it->second;
        }
      }
      log_variant.error("User task %s (ID %d) has no registered variants "
                              "for processors of kind %d and index space %d",
                              name, user_id, kind, index);
#ifdef DEBUG_HIGH_LEVEL
      assert(false);
#endif
      exit(ERROR_UNREGISTERED_VARIANT);
      return variants[0];
    }

    /////////////////////////////////////////////////////////////
    // Task Config Options 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TaskConfigOptions::TaskConfigOptions(bool l /*=false*/,
                                         bool in /*=false*/,
                                         bool idem /*=false*/)
      : leaf(l), inner(in), idempotent(idem)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // ProjectionFunctor 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProjectionFunctor::ProjectionFunctor(Runtime *rt)
      : runtime(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ProjectionFunctor::~ProjectionFunctor(void)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Coloring Serializer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ColoringSerializer::ColoringSerializer(const Coloring &c)
      : coloring(c)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    size_t ColoringSerializer::legion_buffer_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = sizeof(size_t); // number of elements
      for (Coloring::const_iterator it = coloring.begin();
            it != coloring.end(); it++)
      {
        result += sizeof(Color);
        result += 2*sizeof(size_t); // number of each kind of pointer
        result += (it->second.points.size() * sizeof(ptr_t));
        result += (it->second.ranges.size() * 2 * sizeof(ptr_t));
      }
      return result;
    }

    //--------------------------------------------------------------------------
    size_t ColoringSerializer::legion_serialize(void *buffer) const
    //--------------------------------------------------------------------------
    {
      char *target = (char*)buffer; 
      *((size_t*)target) = coloring.size();
      target += sizeof(size_t);
      for (Coloring::const_iterator it = coloring.begin();
            it != coloring.end(); it++)
      {
        *((Color*)target) = it->first;
        target += sizeof(it->first);
        *((size_t*)target) = it->second.points.size();
        target += sizeof(size_t);
        for (std::set<ptr_t>::const_iterator ptr_it = it->second.points.begin();
              ptr_it != it->second.points.end(); ptr_it++)
        {
          *((ptr_t*)target) = *ptr_it;
          target += sizeof(ptr_t);
        }
        *((size_t*)target) = it->second.ranges.size();
        target += sizeof(size_t);
        for (std::set<std::pair<ptr_t,ptr_t> >::const_iterator range_it = 
              it->second.ranges.begin(); range_it != it->second.ranges.end();
              range_it++)
        {
          *((ptr_t*)target) = range_it->first;
          target += sizeof(range_it->first);
          *((ptr_t*)target) = range_it->second;
          target += sizeof(range_it->second);
        }
      }
      return (size_t(target) - size_t(buffer));
    }

    //--------------------------------------------------------------------------
    size_t ColoringSerializer::legion_deserialize(const void *buffer)
    //--------------------------------------------------------------------------
    {
      const char *source = (const char*)buffer;
      size_t num_colors = *((const size_t*)source);
      source += sizeof(num_colors);
      for (unsigned idx = 0; idx < num_colors; idx++)
      {
        Color c = *((const Color*)source);
        source += sizeof(c);
        coloring[c]; // Force coloring to exist even if empty.
        size_t num_points = *((const size_t*)source);
        source += sizeof(num_points);
        for (unsigned p = 0; p < num_points; p++)
        {
          ptr_t ptr = *((const ptr_t*)source);
          source += sizeof(ptr);
          coloring[c].points.insert(ptr);
        }
        size_t num_ranges = *((const size_t*)source);
        source += sizeof(num_ranges);
        for (unsigned r = 0; r < num_ranges; r++)
        {
          ptr_t start = *((const ptr_t*)source);
          source += sizeof(start);
          ptr_t stop = *((const ptr_t*)source);
          source += sizeof(stop);
          coloring[c].ranges.insert(std::pair<ptr_t,ptr_t>(start,stop));
        }
      }
      // Return the number of bytes consumed
      return (size_t(source) - size_t(buffer));
    }

    /////////////////////////////////////////////////////////////
    // Domain Coloring Serializer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DomainColoringSerializer::DomainColoringSerializer(const DomainColoring &d)
      : coloring(d)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    size_t DomainColoringSerializer::legion_buffer_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = sizeof(size_t); // number of elements
      result += (coloring.size() * (sizeof(Color) + sizeof(Domain)));
      return result;
    }

    //--------------------------------------------------------------------------
    size_t DomainColoringSerializer::legion_serialize(void *buffer) const
    //--------------------------------------------------------------------------
    {
      char *target = (char*)buffer;
      *((size_t*)target) = coloring.size();
      target += sizeof(size_t);
      for (DomainColoring::const_iterator it = coloring.begin();
            it != coloring.end(); it++)
      {
        *((Color*)target) = it->first; 
        target += sizeof(it->first);
        *((Domain*)target) = it->second;
        target += sizeof(it->second);
      }
      return (size_t(target) - size_t(buffer));
    }

    //--------------------------------------------------------------------------
    size_t DomainColoringSerializer::legion_deserialize(const void *buffer)
    //--------------------------------------------------------------------------
    {
      const char *source = (const char*)buffer;
      size_t num_elements = *((const size_t*)source);
      source += sizeof(size_t);
      for (unsigned idx = 0; idx < num_elements; idx++)
      {
        Color c = *((const Color*)source);
        source += sizeof(c);
        Domain d = *((const Domain*)source);
        source += sizeof(d);
        coloring[c] = d;
      }
      // Return the number of bytes consumed
      return (size_t(source) - size_t(buffer));
    }

    /////////////////////////////////////////////////////////////
    // High Level Runtime 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Runtime::Runtime(Internal *rt)
      : runtime(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space(Context ctx,
                                                    size_t max_num_elmts)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_space(ctx, max_num_elmts);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space(Context ctx, Domain domain)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_space(ctx, domain);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space(Context ctx, 
                                                const std::set<Domain> &domains)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_space(ctx, domains);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_index_space(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_index_space(ctx, handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(Context ctx,
                                          IndexSpace parent,
                                          const Domain &color_space,
                                          const PointColoring &coloring,
                                          PartitionKind part_kind,
                                          int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_partition(ctx, parent, color_space, coloring,
                                             part_kind, color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(
                                          Context ctx, IndexSpace parent,
                                          const Coloring &coloring,
                                          bool disjoint,
                                          int part_color)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_partition(ctx, parent, coloring, disjoint,
                                             part_color);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(Context ctx,
                                          IndexSpace parent, 
                                          const Domain &color_space,
                                          const DomainPointColoring &coloring,
                                          PartitionKind part_kind, int color)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_partition(ctx, parent, color_space,
                                             coloring, part_kind, color);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(
                                          Context ctx, IndexSpace parent,
                                          Domain color_space,
                                          const DomainColoring &coloring,
                                          bool disjoint, 
                                          int part_color)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_partition(ctx, parent, color_space, coloring,
                                             disjoint, part_color);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(Context ctx,
                                       IndexSpace parent,
                                       const Domain &color_space,
                                       const MultiDomainPointColoring &coloring,
                                       PartitionKind part_kind, int color)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_partition(ctx, parent, color_space,
                                             coloring, part_kind, color);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(
                                          Context ctx, IndexSpace parent,
                                          Domain color_space,
                                          const MultiDomainColoring &coloring,
                                          bool disjoint,
                                          int part_color)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_partition(ctx, parent, color_space, coloring,
                                             disjoint, part_color);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(
                                          Context ctx, IndexSpace parent,
    Accessor::RegionAccessor<Accessor::AccessorType::Generic> field_accessor,
                                          int part_color)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_partition(ctx, parent, field_accessor, 
                                             part_color);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_index_partition(Context ctx, 
                                                   IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_index_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_equal_partition(Context ctx, 
                                                      IndexSpace parent,
                                                      Domain color_space,
                                                      size_t granularity,
                                                      int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      return runtime->create_equal_partition(ctx, parent, color_space,
                                             granularity, color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_weighted_partition(Context ctx,
                                      IndexSpace parent, Domain color_space,
                                      const std::map<DomainPoint,int> &weights,
                                      size_t granularity, int color,
                                      bool allocable)
    //--------------------------------------------------------------------------
    {
      return runtime->create_weighted_partition(ctx, parent, color_space, 
                                                weights, granularity, 
                                                color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_union(Context ctx,
                                    IndexSpace parent, IndexPartition handle1,
                                    IndexPartition handle2, PartitionKind kind,
                                    int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      return runtime->create_partition_by_union(ctx, parent, handle1, handle2, 
                                                kind, color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_intersection(
                                                Context ctx, IndexSpace parent,
                                                IndexPartition handle1, 
                                                IndexPartition handle2,
                                                PartitionKind kind, int color, 
                                                bool allocable)
    //--------------------------------------------------------------------------
    {
      return runtime->create_partition_by_intersection(ctx, parent, handle1,
                                                       handle2, kind,
                                                       color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_difference(
                                                Context ctx, IndexSpace parent,
                                                IndexPartition handle1,
                                                IndexPartition handle2,
                                                PartitionKind kind, int color,
                                                bool allocable)
    //--------------------------------------------------------------------------
    {
      return runtime->create_partition_by_difference(ctx, parent, handle1,
                                                     handle2, kind, color,
                                                     allocable);
    }

    //--------------------------------------------------------------------------
    void Runtime::create_cross_product_partitions(Context ctx,
                                IndexPartition handle1, IndexPartition handle2,
                                std::map<DomainPoint,IndexPartition> &handles,
                                PartitionKind kind, int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      runtime->create_cross_product_partition(ctx, handle1, handle2, handles,
                                              kind, color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_field(Context ctx,
                   LogicalRegion handle, LogicalRegion parent, FieldID fid, 
                   const Domain &color_space, int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      return runtime->create_partition_by_field(ctx, handle, parent, fid,
                                                color_space, color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_image(Context ctx,
                  IndexSpace handle, LogicalPartition projection,
                  LogicalRegion parent, FieldID fid, const Domain &color_space,
                  PartitionKind part_kind, int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      return runtime->create_partition_by_image(ctx, handle, projection,
                                                parent, fid, color_space,
                                                part_kind, color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_preimage(Context ctx,
                  IndexPartition projection, LogicalRegion handle,
                  LogicalRegion parent, FieldID fid, const Domain &color_space,
                  PartitionKind part_kind, int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      return runtime->create_partition_by_preimage(ctx, projection, handle,
                                                   parent, fid, color_space,
                                                   part_kind, color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_pending_partition(Context ctx,
                             IndexSpace parent, const Domain &color_space, 
                             PartitionKind part_kind, int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      return runtime->create_pending_partition(ctx, parent, color_space, 
                                               part_kind, color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_union(Context ctx,
                      IndexPartition parent, const DomainPoint &color,
                      const std::vector<IndexSpace> &handles) 
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_space_union(ctx, parent, color, handles);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_union(Context ctx,
                      IndexPartition parent, const DomainPoint &color,
                      IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_space_union(ctx, parent, color, handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_intersection(Context ctx,
                      IndexPartition parent, const DomainPoint &color,
                      const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_space_intersection(ctx, parent, 
                                                      color, handles);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_intersection(Context ctx,
                      IndexPartition parent, const DomainPoint &color,
                      IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_space_intersection(ctx, parent, 
                                                      color, handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_difference(Context ctx,
          IndexPartition parent, const DomainPoint &color, IndexSpace initial, 
          const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_space_difference(ctx, parent, color, 
                                                    initial, handles);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::get_index_partition(Context ctx, 
                                                IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition(ctx, parent, color);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::get_index_partition(Context ctx,
                                    IndexSpace parent, const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition(ctx, parent, color);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_index_partition(Context ctx, IndexSpace parent,
                                               const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      return runtime->has_index_partition(ctx, parent, color);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_index_subspace(Context ctx, 
                                                  IndexPartition p, Color color)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_subspace(ctx, p, color);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_index_subspace(Context ctx,
                                     IndexPartition p, const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_subspace(ctx, p, color);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_index_subspace(Context ctx, 
                                     IndexPartition p, const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      return runtime->has_index_subspace(ctx, p, color);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_multiple_domains(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->has_multiple_domains(ctx, handle);
    }

    //--------------------------------------------------------------------------
    Domain Runtime::get_index_space_domain(Context ctx, 
                                                    IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_space_domain(ctx, handle);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_domains(Context ctx, 
                                IndexSpace handle, std::vector<Domain> &domains)
    //--------------------------------------------------------------------------
    {
      runtime->get_index_space_domains(ctx, handle, domains);
    }

    //--------------------------------------------------------------------------
    Domain Runtime::get_index_partition_color_space(Context ctx, 
                                                             IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition_color_space(ctx, p);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_partition_colors(Context ctx, 
                                                            IndexSpace sp,
                                                        std::set<Color> &colors)
    //--------------------------------------------------------------------------
    {
      runtime->get_index_space_partition_colors(ctx, sp, colors);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_partition_colors(Context ctx,
                                                            IndexSpace sp,
                                                  std::set<DomainPoint> &colors)
    //--------------------------------------------------------------------------
    {
      runtime->get_index_space_partition_colors(ctx, sp, colors);
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_index_partition_disjoint(Context ctx, 
                                                       IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return runtime->is_index_partition_disjoint(ctx, p);
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_index_space_color(Context ctx, 
                                                  IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_space_color(ctx, handle);
    }

    //--------------------------------------------------------------------------
    DomainPoint Runtime::get_index_space_color_point(Context ctx,
                                                              IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_space_color_point(ctx, handle);
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_index_partition_color(Context ctx,
                                                      IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition_color(ctx, handle);
    }

    //--------------------------------------------------------------------------
    DomainPoint Runtime::get_index_partition_color_point(Context ctx,
                                                          IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition_color_point(ctx, handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_parent_index_space(Context ctx,
                                                        IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_index_space(ctx, handle);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_parent_index_partition(Context ctx,
                                                      IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->has_parent_index_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::get_parent_index_partition(Context ctx,
                                                              IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_index_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    ptr_t Runtime::safe_cast(Context ctx, ptr_t pointer, 
                                      LogicalRegion region)
    //--------------------------------------------------------------------------
    {
      return runtime->safe_cast(ctx, pointer, region);
    }

    //--------------------------------------------------------------------------
    DomainPoint Runtime::safe_cast(Context ctx, DomainPoint point, 
                                            LogicalRegion region)
    //--------------------------------------------------------------------------
    {
      return runtime->safe_cast(ctx, point, region);
    }

    //--------------------------------------------------------------------------
    FieldSpace Runtime::create_field_space(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->create_field_space(ctx);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_field_space(Context ctx, FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_field_space(ctx, handle);
    }

    //--------------------------------------------------------------------------
    size_t Runtime::get_field_size(Context ctx, FieldSpace handle,
                                            FieldID fid)
    //--------------------------------------------------------------------------
    {
      return runtime->get_field_size(ctx, handle, fid);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_field_space_fields(Context ctx, FieldSpace handle,
                                         std::set<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      runtime->get_field_space_fields(ctx, handle, fields);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::create_logical_region(Context ctx, 
                                            IndexSpace index, FieldSpace fields)
    //--------------------------------------------------------------------------
    {
      return runtime->create_logical_region(ctx, index, fields);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_logical_region(Context ctx, 
                                                  LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_logical_region(ctx, handle);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_logical_partition(Context ctx, 
                                                     LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_logical_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition(Context ctx, 
                                    LogicalRegion parent, IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition(ctx, parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_color(
                                    Context ctx, LogicalRegion parent, Color c)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition_by_color(ctx, parent, c);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_color(
                        Context ctx, LogicalRegion parent, const DomainPoint &c)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition_by_color(ctx, parent, c);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_logical_partition_by_color(Context ctx,
                                     LogicalRegion parent, const DomainPoint &c)
    //--------------------------------------------------------------------------
    {
      return runtime->has_logical_partition_by_color(ctx, parent, c);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_tree(
                                            Context ctx, IndexPartition handle, 
                                            FieldSpace fspace, RegionTreeID tid) 
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition_by_tree(ctx, handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion(Context ctx, 
                                    LogicalPartition parent, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_subregion(ctx, parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_color(Context ctx, 
                                             LogicalPartition parent, Color c)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_subregion_by_color(ctx, parent, c);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_color(Context ctx,
                                  LogicalPartition parent, const DomainPoint &c)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_subregion_by_color(ctx, parent, c);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_logical_subregion_by_color(Context ctx,
                                  LogicalPartition parent, const DomainPoint &c)
    //--------------------------------------------------------------------------
    {
      return runtime->has_logical_subregion_by_color(ctx, parent, c);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_tree(Context ctx, 
                        IndexSpace handle, FieldSpace fspace, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_subregion_by_tree(ctx, handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_logical_region_color(Context ctx,
                                                     LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_region_color(ctx, handle);
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_logical_partition_color(Context ctx,
                                                        LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition_color(ctx, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_parent_logical_region(Context ctx,
                                                        LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_logical_region(ctx, handle);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_parent_logical_partition(Context ctx,
                                                        LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return runtime->has_parent_logical_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_parent_logical_partition(Context ctx,
                                                           LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_logical_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    IndexAllocator Runtime::create_index_allocator(Context ctx, 
                                                            IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_allocator(ctx, handle);
    }

    //--------------------------------------------------------------------------
    FieldAllocator Runtime::create_field_allocator(Context ctx, 
                                                            FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->create_field_allocator(ctx, handle);
    }

    //--------------------------------------------------------------------------
    ArgumentMap Runtime::create_argument_map(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->create_argument_map(ctx);
    }

    //--------------------------------------------------------------------------
    Future Runtime::execute_task(Context ctx, 
                                          const TaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return runtime->execute_task(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    FutureMap Runtime::execute_index_space(Context ctx, 
                                                  const IndexLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return runtime->execute_index_space(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    Future Runtime::execute_index_space(Context ctx, 
                            const IndexLauncher &launcher, ReductionOpID redop)
    //--------------------------------------------------------------------------
    {
      return runtime->execute_index_space(ctx, launcher, redop);
    }

    //--------------------------------------------------------------------------
    Future Runtime::execute_task(Context ctx, 
                        Processor::TaskFuncID task_id,
                        const std::vector<IndexSpaceRequirement> &indexes,
                        const std::vector<FieldSpaceRequirement> &fields,
                        const std::vector<RegionRequirement> &regions,
                        const TaskArgument &arg, 
                        const Predicate &predicate,
                        MapperID id, 
                        MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return runtime->execute_task(ctx, task_id, indexes, fields, regions,
                                   arg, predicate, id, tag);
    }

    //--------------------------------------------------------------------------
    FutureMap Runtime::execute_index_space(Context ctx, 
                        Processor::TaskFuncID task_id,
                        const Domain domain,
                        const std::vector<IndexSpaceRequirement> &indexes,
                        const std::vector<FieldSpaceRequirement> &fields,
                        const std::vector<RegionRequirement> &regions,
                        const TaskArgument &global_arg, 
                        const ArgumentMap &arg_map,
                        const Predicate &predicate,
                        bool must_parallelism, 
                        MapperID id, 
                        MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return runtime->execute_index_space(ctx, task_id, domain, indexes,
                                          fields, regions, global_arg,
                                          arg_map, predicate,
                                          must_parallelism, id, tag);
    }


    //--------------------------------------------------------------------------
    Future Runtime::execute_index_space(Context ctx, 
                        Processor::TaskFuncID task_id,
                        const Domain domain,
                        const std::vector<IndexSpaceRequirement> &indexes,
                        const std::vector<FieldSpaceRequirement> &fields,
                        const std::vector<RegionRequirement> &regions,
                        const TaskArgument &global_arg, 
                        const ArgumentMap &arg_map,
                        ReductionOpID reduction, 
                        const TaskArgument &initial_value,
                        const Predicate &predicate,
                        bool must_parallelism, 
                        MapperID id, 
                        MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return runtime->execute_index_space(ctx, task_id, domain, indexes,
                                          fields, regions, global_arg, arg_map,
                                          reduction, initial_value, predicate,
                                          must_parallelism, id, tag);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion Runtime::map_region(Context ctx, 
                                                const InlineLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return runtime->map_region(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion Runtime::map_region(Context ctx, 
                    const RegionRequirement &req, MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return runtime->map_region(ctx, req, id, tag);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion Runtime::map_region(Context ctx, unsigned idx, 
                                                  MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return runtime->map_region(ctx, idx, id, tag);
    }

    //--------------------------------------------------------------------------
    void Runtime::remap_region(Context ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      runtime->remap_region(ctx, region);
    }

    //--------------------------------------------------------------------------
    void Runtime::unmap_region(Context ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      runtime->unmap_region(ctx, region);
    }

    //--------------------------------------------------------------------------
    void Runtime::unmap_all_regions(Context ctx)
    //--------------------------------------------------------------------------
    {
      runtime->unmap_all_regions(ctx);
    }

    //--------------------------------------------------------------------------
    void Runtime::fill_field(Context ctx, LogicalRegion handle,
                                      LogicalRegion parent, FieldID fid,
                                      const void *value, size_t value_size,
                                      Predicate pred)
    //--------------------------------------------------------------------------
    {
      runtime->fill_field(ctx, handle, parent, fid, value, value_size, pred);
    }

    //--------------------------------------------------------------------------
    void Runtime::fill_field(Context ctx, LogicalRegion handle,
                                      LogicalRegion parent, FieldID fid,
                                      Future f, Predicate pred)
    //--------------------------------------------------------------------------
    {
      runtime->fill_field(ctx, handle, parent, fid, f, pred);
    }

    //--------------------------------------------------------------------------
    void Runtime::fill_fields(Context ctx, LogicalRegion handle,
                                       LogicalRegion parent,
                                       const std::set<FieldID> &fields,
                                       const void *value, size_t size,
                                       Predicate pred)
    //--------------------------------------------------------------------------
    {
      runtime->fill_fields(ctx, handle, parent, fields, value, size, pred);
    }

    //--------------------------------------------------------------------------
    void Runtime::fill_fields(Context ctx, LogicalRegion handle,
                                       LogicalRegion parent, 
                                       const std::set<FieldID> &fields,
                                       Future f, Predicate pred)
    //--------------------------------------------------------------------------
    {
      runtime->fill_fields(ctx, handle, parent, fields, f, pred);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion Runtime::attach_hdf5(Context ctx, 
                                                 const char *file_name,
                                                 LogicalRegion handle,
                                                 LogicalRegion parent,
                                 const std::map<FieldID,const char*> &field_map,
                                                 LegionFileMode mode)
    //--------------------------------------------------------------------------
    {
      return runtime->attach_hdf5(ctx, file_name, handle, 
                                  parent, field_map, mode);
    }

    //--------------------------------------------------------------------------
    void Runtime::detach_hdf5(Context ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      runtime->detach_hdf5(ctx, region);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion Runtime::attach_file(Context ctx,
                                                 const char *file_name,
                                                 LogicalRegion handle,
                                                 LogicalRegion parent,
                                 const std::vector<FieldID> &field_vec,
                                                 LegionFileMode mode)
    //--------------------------------------------------------------------------
    {
      return runtime->attach_file(ctx, file_name, handle,
                                  parent, field_vec, mode);
    }

    //--------------------------------------------------------------------------
    void Runtime::detach_file(Context ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      runtime->detach_file(ctx, region);
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_copy_operation(Context ctx, 
                                                const CopyLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      runtime->issue_copy_operation(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    Predicate Runtime::create_predicate(Context ctx, const Future &f)
    //--------------------------------------------------------------------------
    {
      return runtime->create_predicate(ctx, f);
    }

    //--------------------------------------------------------------------------
    Predicate Runtime::predicate_not(Context ctx, const Predicate &p) 
    //--------------------------------------------------------------------------
    {
      return runtime->predicate_not(ctx, p);
    }

    //--------------------------------------------------------------------------
    Predicate Runtime::predicate_and(Context ctx, 
                                       const Predicate &p1, const Predicate &p2) 
    //--------------------------------------------------------------------------
    {
      return runtime->predicate_and(ctx, p1, p2);
    }

    //--------------------------------------------------------------------------
    Predicate Runtime::predicate_or(Context ctx,
                                       const Predicate &p1, const Predicate &p2)  
    //--------------------------------------------------------------------------
    {
      return runtime->predicate_or(ctx, p1, p2);
    }

    //--------------------------------------------------------------------------
    Lock Runtime::create_lock(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->create_lock(ctx);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_lock(Context ctx, Lock l)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_lock(ctx, l);
    }

    //--------------------------------------------------------------------------
    Grant Runtime::acquire_grant(Context ctx,
                                      const std::vector<LockRequest> &requests)
    //--------------------------------------------------------------------------
    {
      return runtime->acquire_grant(ctx, requests);
    }

    //--------------------------------------------------------------------------
    void Runtime::release_grant(Context ctx, Grant grant)
    //--------------------------------------------------------------------------
    {
      runtime->release_grant(ctx, grant);
    }

    //--------------------------------------------------------------------------
    PhaseBarrier Runtime::create_phase_barrier(Context ctx, 
                                                        unsigned arrivals)
    //--------------------------------------------------------------------------
    {
      return runtime->create_phase_barrier(ctx, arrivals);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_phase_barrier(Context ctx, PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_phase_barrier(ctx, pb);
    }

    //--------------------------------------------------------------------------
    PhaseBarrier Runtime::advance_phase_barrier(Context ctx, 
                                                         PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      return runtime->advance_phase_barrier(ctx, pb);
    }

    //--------------------------------------------------------------------------
    DynamicCollective Runtime::create_dynamic_collective(Context ctx,
                                                        unsigned arrivals,
                                                        ReductionOpID redop,
                                                        const void *init_value,
                                                        size_t init_size)
    //--------------------------------------------------------------------------
    {
      return runtime->create_dynamic_collective(ctx, arrivals, redop,
                                                init_value, init_size);
    }
    
    //--------------------------------------------------------------------------
    void Runtime::destroy_dynamic_collective(Context ctx, 
                                                      DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_dynamic_collective(ctx, dc);
    }

    //--------------------------------------------------------------------------
    void Runtime::arrive_dynamic_collective(Context ctx,
                                                     DynamicCollective dc,
                                                     const void *buffer,
                                                     size_t size, 
                                                     unsigned count)
    //--------------------------------------------------------------------------
    {
      runtime->arrive_dynamic_collective(ctx, dc, buffer, size, count);
    }

    //--------------------------------------------------------------------------
    void Runtime::defer_dynamic_collective_arrival(Context ctx,
                                                      DynamicCollective dc,
                                                      Future f, unsigned count)
    //--------------------------------------------------------------------------
    {
      runtime->defer_dynamic_collective_arrival(ctx, dc, f, count);
    }

    //--------------------------------------------------------------------------
    Future Runtime::get_dynamic_collective_result(Context ctx,
                                                           DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
      return runtime->get_dynamic_collective_result(ctx, dc);
    }

    //--------------------------------------------------------------------------
    DynamicCollective Runtime::advance_dynamic_collective(Context ctx,
                                                           DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
      return runtime->advance_dynamic_collective(ctx, dc);
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_acquire(Context ctx,
                                         const AcquireLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      runtime->issue_acquire(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_release(Context ctx,
                                         const ReleaseLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      runtime->issue_release(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_mapping_fence(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->issue_mapping_fence(ctx);
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_execution_fence(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->issue_execution_fence(ctx);
    }

    //--------------------------------------------------------------------------
    void Runtime::begin_trace(Context ctx, TraceID tid)
    //--------------------------------------------------------------------------
    {
      runtime->begin_trace(ctx, tid);
    }

    //--------------------------------------------------------------------------
    void Runtime::end_trace(Context ctx, TraceID tid)
    //--------------------------------------------------------------------------
    {
      runtime->end_trace(ctx, tid);
    }

    //--------------------------------------------------------------------------
    void Runtime::complete_frame(Context ctx)
    //--------------------------------------------------------------------------
    {
      runtime->complete_frame(ctx);
    }

    //--------------------------------------------------------------------------
    FutureMap Runtime::execute_must_epoch(Context ctx,
                                              const MustEpochLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return runtime->execute_must_epoch(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    int Runtime::get_tunable_value(Context ctx, TunableID tid,
                                            MapperID mid, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return runtime->get_tunable_value(ctx, tid, mid, tag);
    }

    //--------------------------------------------------------------------------
    Future Runtime::get_current_time(Context ctx, Future precondition)
    //--------------------------------------------------------------------------
    {
      return runtime->get_current_time(ctx, precondition);
    }

    //--------------------------------------------------------------------------
    Future Runtime::get_current_time_in_microseconds(Context ctx, Future pre)
    //--------------------------------------------------------------------------
    {
      return runtime->get_current_time_in_microseconds(ctx, pre);
    }

    //--------------------------------------------------------------------------
    Future Runtime::get_current_time_in_nanoseconds(Context ctx, Future pre)
    //--------------------------------------------------------------------------
    {
      return runtime->get_current_time_in_nanoseconds(ctx, pre);
    }

    //--------------------------------------------------------------------------
    /*static*/ long long Runtime::get_zero_time(void)
    //--------------------------------------------------------------------------
    {
      return Realm::Clock::get_zero_time();
    }

    //--------------------------------------------------------------------------
    Mapper* Runtime::get_mapper(Context ctx, MapperID id,
                                         Processor target)
    //--------------------------------------------------------------------------
    {
      return runtime->get_mapper(ctx, id, target);
    }

    //--------------------------------------------------------------------------
    Processor Runtime::get_executing_processor(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->get_executing_processor(ctx);
    }

    //--------------------------------------------------------------------------
    void Runtime::raise_region_exception(Context ctx, 
                                                  PhysicalRegion region,
                                                  bool nuclear)
    //--------------------------------------------------------------------------
    {
      runtime->raise_region_exception(ctx, region, nuclear);
    }

    //--------------------------------------------------------------------------
    const std::map<int,AddressSpace>& 
                                Runtime::find_forward_MPI_mapping(void)
    //--------------------------------------------------------------------------
    {
      return runtime->find_forward_MPI_mapping();
    }

    //--------------------------------------------------------------------------
    const std::map<AddressSpace,int>&
                                Runtime::find_reverse_MPI_mapping(void)
    //--------------------------------------------------------------------------
    {
      return runtime->find_reverse_MPI_mapping();
    }

    //--------------------------------------------------------------------------
    MapperID Runtime::generate_dynamic_mapper_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_mapper_id();
    }

    //--------------------------------------------------------------------------
    /*static*/ MapperID Runtime::generate_static_mapper_id(void)
    //--------------------------------------------------------------------------
    {
      return Internal::generate_static_mapper_id();
    }

    //--------------------------------------------------------------------------
    void Runtime::add_mapper(MapperID map_id, Mapper *mapper, 
                                      Processor proc)
    //--------------------------------------------------------------------------
    {
      runtime->add_mapper(map_id, mapper, proc);
    }

    //--------------------------------------------------------------------------
    void Runtime::replace_default_mapper(Mapper *mapper, 
                                                  Processor proc)
    //--------------------------------------------------------------------------
    {
      runtime->replace_default_mapper(mapper, proc);
    }

    //--------------------------------------------------------------------------
    void Runtime::register_projection_functor(ProjectionID pid,
                                                       ProjectionFunctor *func)
    //--------------------------------------------------------------------------
    {
      runtime->register_projection_functor(pid, func);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(TaskID task_id, SemanticTag tag,
                                   const void *buffer, size_t size, bool is_mut)
    //--------------------------------------------------------------------------
    {
      runtime->attach_semantic_information(task_id, tag, buffer, size, is_mut);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(IndexSpace handle,
                                                       SemanticTag tag,
                                                       const void *buffer,
                                                       size_t size, bool is_mut)
    //--------------------------------------------------------------------------
    {
      runtime->attach_semantic_information(handle, tag, buffer, size, is_mut);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(IndexPartition handle,
                                                       SemanticTag tag,
                                                       const void *buffer,
                                                       size_t size, bool is_mut)
    //--------------------------------------------------------------------------
    {
      runtime->attach_semantic_information(handle, tag, buffer, size, is_mut);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(FieldSpace handle,
                                                       SemanticTag tag,
                                                       const void *buffer,
                                                       size_t size, bool is_mut)
    //--------------------------------------------------------------------------
    {
      runtime->attach_semantic_information(handle, tag, buffer, size, is_mut);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(FieldSpace handle,
                                                       FieldID fid,
                                                       SemanticTag tag,
                                                       const void *buffer,
                                                       size_t size, bool is_mut)
    //--------------------------------------------------------------------------
    {
      runtime->attach_semantic_information(handle, fid, tag, buffer, 
                                           size, is_mut);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(LogicalRegion handle,
                                                       SemanticTag tag,
                                                       const void *buffer,
                                                       size_t size, bool is_mut)
    //--------------------------------------------------------------------------
    {
      runtime->attach_semantic_information(handle, tag, buffer, size, is_mut);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(LogicalPartition handle,
                                                       SemanticTag tag,
                                                       const void *buffer,
                                                       size_t size, bool is_mut)
    //--------------------------------------------------------------------------
    {
      runtime->attach_semantic_information(handle, tag, buffer, size, is_mut);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_name(TaskID task_id, const char *name, bool is_mutable)
    //--------------------------------------------------------------------------
    {
      Runtime::attach_semantic_information(task_id,
          NAME_SEMANTIC_TAG, name, strlen(name) + 1, is_mutable);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_name(IndexSpace handle, const char *name, bool is_mut)
    //--------------------------------------------------------------------------
    {
      Runtime::attach_semantic_information(handle,
          NAME_SEMANTIC_TAG, name, strlen(name) + 1, is_mut);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_name(IndexPartition handle, const char *name, bool ism)
    //--------------------------------------------------------------------------
    {
      Runtime::attach_semantic_information(handle,
          NAME_SEMANTIC_TAG, name, strlen(name) + 1, ism);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_name(FieldSpace handle, const char *name, bool is_mut)
    //--------------------------------------------------------------------------
    {
      Runtime::attach_semantic_information(handle,
          NAME_SEMANTIC_TAG, name, strlen(name) + 1, is_mut);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_name(FieldSpace handle,
                                       FieldID fid,
                                       const char *name, bool is_mutable)
    //--------------------------------------------------------------------------
    {
      Runtime::attach_semantic_information(handle, fid,
          NAME_SEMANTIC_TAG, name, strlen(name) + 1, is_mutable);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_name(LogicalRegion handle, const char *name, bool ism)
    //--------------------------------------------------------------------------
    {
      Runtime::attach_semantic_information(handle,
          NAME_SEMANTIC_TAG, name, strlen(name) + 1, ism);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_name(LogicalPartition handle, const char *name, bool m)
    //--------------------------------------------------------------------------
    {
      Runtime::attach_semantic_information(handle,
          NAME_SEMANTIC_TAG, name, strlen(name) + 1, m);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_semantic_information(TaskID task_id, SemanticTag tag,
                                              const void *&result, size_t &size)
    //--------------------------------------------------------------------------
    {
      runtime->retrieve_semantic_information(task_id, tag, result, size);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_semantic_information(IndexSpace handle,
                                                         SemanticTag tag,
                                                         const void *&result,
                                                         size_t &size)
    //--------------------------------------------------------------------------
    {
      runtime->retrieve_semantic_information(handle, tag, result, size);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_semantic_information(IndexPartition handle,
                                                         SemanticTag tag,
                                                         const void *&result,
                                                         size_t &size)
    //--------------------------------------------------------------------------
    {
      runtime->retrieve_semantic_information(handle, tag, result, size);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_semantic_information(FieldSpace handle,
                                                         SemanticTag tag,
                                                         const void *&result,
                                                         size_t &size)
    //--------------------------------------------------------------------------
    {
      runtime->retrieve_semantic_information(handle, tag, result, size);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_semantic_information(FieldSpace handle,
                                                         FieldID fid,
                                                         SemanticTag tag,
                                                         const void *&result,
                                                         size_t &size)
    //--------------------------------------------------------------------------
    {
      runtime->retrieve_semantic_information(handle, fid, tag, result, size);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_semantic_information(LogicalRegion handle,
                                                         SemanticTag tag,
                                                         const void *&result,
                                                         size_t &size)
    //--------------------------------------------------------------------------
    {
      runtime->retrieve_semantic_information(handle, tag, result, size);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_semantic_information(LogicalPartition part,
                                                         SemanticTag tag,
                                                         const void *&result,
                                                         size_t &size)
    //--------------------------------------------------------------------------
    {
      runtime->retrieve_semantic_information(part, tag, result, size);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_name(TaskID task_id, const char *&result)
    //--------------------------------------------------------------------------
    {
      const void* dummy_ptr; size_t dummy_size;
      Runtime::retrieve_semantic_information(task_id, NAME_SEMANTIC_TAG,
                                             dummy_ptr, dummy_size);
      result = reinterpret_cast<const char*>(dummy_ptr);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_name(IndexSpace handle, const char *&result)
    //--------------------------------------------------------------------------
    {
      const void* dummy_ptr; size_t dummy_size;
      Runtime::retrieve_semantic_information(handle,
          NAME_SEMANTIC_TAG, dummy_ptr, dummy_size);
      result = reinterpret_cast<const char*>(dummy_ptr);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_name(IndexPartition handle,
                                         const char *&result)
    //--------------------------------------------------------------------------
    {
      const void* dummy_ptr; size_t dummy_size;
      Runtime::retrieve_semantic_information(handle,
          NAME_SEMANTIC_TAG, dummy_ptr, dummy_size);
      result = reinterpret_cast<const char*>(dummy_ptr);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_name(FieldSpace handle, const char *&result)
    //--------------------------------------------------------------------------
    {
      const void* dummy_ptr; size_t dummy_size;
      Runtime::retrieve_semantic_information(handle,
          NAME_SEMANTIC_TAG, dummy_ptr, dummy_size);
      result = reinterpret_cast<const char*>(dummy_ptr);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_name(FieldSpace handle,
                                         FieldID fid,
                                         const char *&result)
    //--------------------------------------------------------------------------
    {
      const void* dummy_ptr; size_t dummy_size;
      Runtime::retrieve_semantic_information(handle, fid,
          NAME_SEMANTIC_TAG, dummy_ptr, dummy_size);
      result = reinterpret_cast<const char*>(dummy_ptr);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_name(LogicalRegion handle,
                                         const char *&result)
    //--------------------------------------------------------------------------
    {
      const void* dummy_ptr; size_t dummy_size;
      Runtime::retrieve_semantic_information(handle,
          NAME_SEMANTIC_TAG, dummy_ptr, dummy_size);
      result = reinterpret_cast<const char*>(dummy_ptr);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_name(LogicalPartition part,
                                         const char *&result)
    //--------------------------------------------------------------------------
    {
      const void* dummy_ptr; size_t dummy_size;
      Runtime::retrieve_semantic_information(part,
          NAME_SEMANTIC_TAG, dummy_ptr, dummy_size);
      result = reinterpret_cast<const char*>(dummy_ptr);
    }

    //--------------------------------------------------------------------------
    FieldID Runtime::allocate_field(Context ctx, FieldSpace space,
                                             size_t field_size, FieldID fid,
                                             bool local, CustomSerdezID sd_id)
    //--------------------------------------------------------------------------
    {
      return runtime->allocate_field(ctx, space, field_size, fid, local, sd_id);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_field(Context ctx, FieldSpace sp, FieldID fid)
    //--------------------------------------------------------------------------
    {
      runtime->free_field(ctx, sp, fid);
    }

    //--------------------------------------------------------------------------
    void Runtime::allocate_fields(Context ctx, FieldSpace space,
                                           const std::vector<size_t> &sizes,
                                         std::vector<FieldID> &resulting_fields,
                                         bool local, CustomSerdezID _id)
    //--------------------------------------------------------------------------
    {
      runtime->allocate_fields(ctx, space, sizes, resulting_fields, local, _id);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_fields(Context ctx, FieldSpace space,
                                       const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      runtime->free_fields(ctx, space, to_free);
    }

    //--------------------------------------------------------------------------
    const std::vector<PhysicalRegion>& Runtime::begin_task(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->begin_task(ctx);
    }

    //--------------------------------------------------------------------------
    const std::vector<PhysicalRegion>& Runtime::begin_inline_task(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->begin_inline_task(ctx);
    }

    //--------------------------------------------------------------------------
    void Runtime::end_task(Context ctx, const void *result, 
                                    size_t result_size, bool owned /*= false*/)
    //--------------------------------------------------------------------------
    {
      runtime->end_task(ctx, result, result_size, owned);
    }

    //--------------------------------------------------------------------------
    void Runtime::end_inline_task(Context ctx, const void *result,
                                  size_t result_size, bool owned /*= false*/)
    //--------------------------------------------------------------------------
    {
      runtime->end_inline_task(ctx, result, result_size, owned);
    }

    //--------------------------------------------------------------------------
    Future Runtime::from_value(const void *value, 
                                        size_t value_size, bool owned)
    //--------------------------------------------------------------------------
    {
      Future result = runtime->help_create_future();
      // Set the future result
      result.impl->set_result(value, value_size, owned);
      // Complete the future right away so that it is always complete
      result.impl->complete_future();
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ int Runtime::start(int argc, char **argv, 
                                           bool background)
    //--------------------------------------------------------------------------
    {
      return Internal::start(argc, argv, background);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::wait_for_shutdown(void)
    //--------------------------------------------------------------------------
    {
      Internal::wait_for_shutdown();
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::set_top_level_task_id(
                                                  Processor::TaskFuncID top_id)
    //--------------------------------------------------------------------------
    {
      Internal::set_top_level_task_id(top_id);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::configure_MPI_interoperability(int rank)
    //--------------------------------------------------------------------------
    {
      Internal::configure_MPI_interoperability(rank);
    }

    //--------------------------------------------------------------------------
    /*static*/ MPILegionHandshake Runtime::create_handshake(
                                                        bool init_in_MPI,
                                                        int mpi_participants,
                                                        int legion_participants)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(mpi_participants > 0);
      assert(legion_participants > 0);
#endif
      return MPILegionHandshake(
          legion_new<MPILegionHandshake::Impl>(init_in_MPI,
                                       mpi_participants, legion_participants));
    }

    //--------------------------------------------------------------------------
    /*static*/ const ReductionOp* Runtime::get_reduction_op(
                                                        ReductionOpID redop_id)
    //--------------------------------------------------------------------------
    {
      return Internal::get_reduction_op(redop_id);
    }

    //--------------------------------------------------------------------------
    /*static*/ const SerdezOp* Runtime::get_serdez_op(CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
      return Internal::get_serdez_op(serdez_id);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::set_registration_callback(
                                            RegistrationCallbackFnptr callback)
    //--------------------------------------------------------------------------
    {
      Internal::set_registration_callback(callback);
    }

    //--------------------------------------------------------------------------
    /*static*/ const InputArgs& Runtime::get_input_args(void)
    //--------------------------------------------------------------------------
    {
      return Internal::get_input_args();
    }

    //--------------------------------------------------------------------------
    /*static*/ Runtime* Runtime::get_runtime(Processor p)
    //--------------------------------------------------------------------------
    {
      return Internal::get_runtime(p)->high_level;
    }

    //--------------------------------------------------------------------------
    /*static*/ ReductionOpTable& Runtime::get_reduction_table(void)
    //--------------------------------------------------------------------------
    {
      return Internal::get_reduction_table();
    }

    //--------------------------------------------------------------------------
    /*static*/ SerdezOpTable& Runtime::get_serdez_table(void)
    //--------------------------------------------------------------------------
    {
      return Internal::get_serdez_table();
    }

    /*static*/ SerdezRedopTable& Runtime::get_serdez_redop_table(void)
    //--------------------------------------------------------------------------
    {
      return Internal::get_serdez_redop_table();
    }

    //--------------------------------------------------------------------------
    /*static*/ ProjectionID Runtime::
      register_region_projection_function(ProjectionID handle, void *func_ptr)
    //--------------------------------------------------------------------------
    {
      return Internal::register_region_projection_function(handle, 
                                                                func_ptr);
    }

    //--------------------------------------------------------------------------
    /*static*/ ProjectionID Runtime::
      register_partition_projection_function(ProjectionID handle, 
                                             void *func_ptr)
    //--------------------------------------------------------------------------
    {
      return Internal::register_partition_projection_function(handle,
                                                                   func_ptr);
    }

    //--------------------------------------------------------------------------
    TaskID Runtime::generate_dynamic_task_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_task_id();
    }

    //--------------------------------------------------------------------------
    /*static*/ TaskID Runtime::generate_static_task_id(void)
    //--------------------------------------------------------------------------
    {
      return Internal::generate_static_task_id();
    }

    //--------------------------------------------------------------------------
    VariantID Runtime::register_variant(const TaskVariantRegistrar &registrar,
                  bool has_return, const void *user_data, size_t user_data_size,
                  CodeDescriptor *realm, CodeDescriptor *indesc)
    //--------------------------------------------------------------------------
    {
      return runtime->register_variant(registrar, user_data, user_data_size,
                                       realm, indesc, has_return);
    }
    
    //--------------------------------------------------------------------------
    /*static*/ VariantID Runtime::preregister_variant(
                                  const TaskVariantRegistrar &registrar,
                                  const void *user_data, size_t user_data_size,
                                  CodeDescriptor *realm, CodeDescriptor *indesc,
                                  bool has_return, const char *task_name, 
                                  bool check_task_id)
    //--------------------------------------------------------------------------
    {
      return Internal::preregister_variant(registrar, user_data, user_data_size,
                           realm, indesc, has_return, task_name, check_task_id);
    } 

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::enable_profiling(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::disable_profiling(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::dump_profiling(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LayoutConstraintID Runtime::register_layout(
                                     const LayoutConstraintRegistrar &registrar)
    //--------------------------------------------------------------------------
    {
      return runtime->register_layout(registrar, AUTO_GENERATE_ID);
    }

    //--------------------------------------------------------------------------
    void Runtime::release_layout(LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      runtime->release_layout(layout_id, runtime->address_space/*local*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ LayoutConstraintID Runtime::preregister_layout(
                                     const LayoutConstraintRegistrar &registrar,
                                     LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      return Internal::preregister_layout(registrar, layout_id);
    }

    //--------------------------------------------------------------------------
    FieldSpace Runtime::get_layout_constraint_field_space(
                                                   LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      return runtime->get_layout_constraint_field_space(layout_id);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_layout_constraints(LayoutConstraintID layout_id,
                                        LayoutConstraintSet &layout_constraints)   
    //--------------------------------------------------------------------------
    {
      runtime->get_layout_constraints(layout_id, layout_constraints);
    }

    //--------------------------------------------------------------------------
    const char* Runtime::get_layout_constraints_name(LayoutConstraintID id)
    //--------------------------------------------------------------------------
    {
      return runtime->get_layout_constraints_name(id);
    }

    /////////////////////////////////////////////////////////////
    // Mapper 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void Mapper::send_message(Processor target, 
                              const void *message, size_t length)
    //--------------------------------------------------------------------------
    {
      runtime->runtime->handle_mapper_send_message(this, target, 
                                                   message, length);
    }

    //--------------------------------------------------------------------------
    void Mapper::broadcast_message(const void *message, size_t length, 
                                   int radix /*=4*/)
    //--------------------------------------------------------------------------
    {
      runtime->runtime->handle_mapper_broadcast(this, message, length, radix);
    }

    //--------------------------------------------------------------------------
    MapperEvent Mapper::launch_mapper_task(Processor::TaskFuncID tid,
                                           const TaskArgument &arg)
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->launch_mapper_task(this, tid, arg);
    }

    //--------------------------------------------------------------------------
    void Mapper::defer_mapper_call(MapperEvent event)
    //--------------------------------------------------------------------------
    {
      runtime->runtime->defer_mapper_call(this, event);
    }

    //--------------------------------------------------------------------------
    MapperEvent Mapper::merge_mapper_events(const std::set<MapperEvent> &events)
    //--------------------------------------------------------------------------
    {
      return Event::merge_events(events);
    }

    //--------------------------------------------------------------------------
    IndexPartition Mapper::get_index_partition(IndexSpace parent, 
                                               Color color) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_index_partition(parent, color);
    }

    //--------------------------------------------------------------------------
    IndexSpace Mapper::get_index_subspace(IndexPartition p, Color c) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_index_subspace(p, c);
    }

    //--------------------------------------------------------------------------
    IndexSpace Mapper::get_index_subspace(IndexPartition p, 
                                          const DomainPoint &color) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_index_subspace(p, color);
    }

    //--------------------------------------------------------------------------
    bool Mapper::has_multiple_domains(IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->has_multiple_domains(handle);
    }

    //--------------------------------------------------------------------------
    Domain Mapper::get_index_space_domain(IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_index_space_domain(handle);
    }

    //--------------------------------------------------------------------------
    void Mapper::get_index_space_domains(IndexSpace handle,
                                         std::vector<Domain> &domains) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_index_space_domains(handle, domains);
    }

    //--------------------------------------------------------------------------
    Domain Mapper::get_index_partition_color_space(IndexPartition p) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_index_partition_color_space(p);
    }

    //--------------------------------------------------------------------------
    void Mapper::get_index_space_partition_colors(
                              IndexSpace handle, std::set<Color> &colors) const
    //--------------------------------------------------------------------------
    {
      runtime->runtime->get_index_space_partition_colors(handle, colors);
    }

    //--------------------------------------------------------------------------
    bool Mapper::is_index_partition_disjoint(IndexPartition p) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->is_index_partition_disjoint(p);
    }

    //--------------------------------------------------------------------------
    Color Mapper::get_index_space_color(IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_index_space_color(handle);
    }

    //--------------------------------------------------------------------------
    Color Mapper::get_index_partition_color(IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_index_partition_color(handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace Mapper::get_parent_index_space(IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_parent_index_space(handle);
    }

    //--------------------------------------------------------------------------
    bool Mapper::has_parent_index_partition(IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->has_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition Mapper::get_parent_index_partition(IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    size_t Mapper::get_field_size(FieldSpace handle, FieldID fid) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_field_size(handle, fid);
    }

    //--------------------------------------------------------------------------
    void Mapper::get_field_space_fields(FieldSpace handle, 
                                        std::set<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      runtime->runtime->get_field_space_fields(handle, fields);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Mapper::get_logical_partition(LogicalRegion parent,
                                                   IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_logical_partition(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Mapper::get_logical_partition_by_color(LogicalRegion par,
                                                            Color color) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_logical_partition_by_color(par, color);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Mapper::get_logical_partition_by_tree(
                                                        IndexPartition part,
                                                        FieldSpace fspace, 
                                                        RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_logical_partition_by_tree(part, fspace, tid);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Mapper::get_logical_subregion(LogicalPartition parent,
                                                IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_logical_subregion(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Mapper::get_logical_subregion_by_color(LogicalPartition par,
                                                         Color color) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_logical_subregion_by_color(par, color);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Mapper::get_logical_subregion_by_tree(IndexSpace handle,
                                                        FieldSpace fspace,
                                                        RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_logical_subregion_by_tree(handle, 
                                                             fspace, tid);
    }

    //--------------------------------------------------------------------------
    Color Mapper::get_logical_region_color(LogicalRegion handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_logical_region_color(handle);
    }

    //--------------------------------------------------------------------------
    Color Mapper::get_logical_partition_color(LogicalPartition handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_logical_partition_color(handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Mapper::get_parent_logical_region(LogicalPartition part) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_parent_logical_region(part);
    }
    
    //--------------------------------------------------------------------------
    bool Mapper::has_parent_logical_partition(LogicalRegion handle) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->has_parent_logical_partition(handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Mapper::get_parent_logical_partition(LogicalRegion r) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->get_parent_logical_partition(r);
    }

    //--------------------------------------------------------------------------
    size_t Mapper::sample_allocated_space(Memory mem) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->sample_allocated_space(mem);
    }

    //--------------------------------------------------------------------------
    size_t Mapper::sample_free_space(Memory mem) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->sample_free_space(mem);
    }

    //--------------------------------------------------------------------------
    unsigned Mapper::sample_allocated_instances(Memory mem) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->sample_allocated_instances(mem);
    }

    //--------------------------------------------------------------------------
    unsigned Mapper::sample_unmapped_tasks(Processor proc) const
    //--------------------------------------------------------------------------
    {
      return runtime->runtime->sample_unmapped_tasks(proc, 
                                                     const_cast<Mapper*>(this));
    } 

  }; // namespace HighLevel
}; // namespace LegionRuntime

// EOF

