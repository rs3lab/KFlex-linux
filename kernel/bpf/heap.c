// SPDX-License-Identifier: GPL-2.0-only
#include <linux/bpf.h>
#include <linux/btf.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/filter.h>
#include <linux/perf_event.h>
#include <uapi/linux/btf.h>
#include <linux/rcupdate_trace.h>
#include <linux/btf_ids.h>

#define HEAP_CREATE_FLAG_MASK \
	(BPF_F_NUMA_NODE | BPF_F_MMAPABLE | BPF_F_ACCESS_MASK | BPF_F_HEAP_TRANS)

struct bpf_heap {
	struct bpf_map map;
	u32 elem_size;
	u32 index_mask;
	void *value __aligned(8);
};

/* Called from syscall */
static int heap_map_alloc_check(union bpf_attr *attr)
{
	/* check sanity of attributes */
	if (attr->max_entries == 0 || attr->key_size != 4 ||
	    attr->value_size == 0 ||
	    attr->map_flags & ~HEAP_CREATE_FLAG_MASK ||
	    !bpf_map_flags_access_ok(attr->map_flags))
		return -EINVAL;

	/* avoid overflow on round_up(map->value_size) */
	if (attr->value_size > INT_MAX)
		return -E2BIG;

	return 0;
}

static struct bpf_map *heap_map_alloc(union bpf_attr *attr)
{
	int numa_node = bpf_map_attr_numa_node(attr);
	bool bypass_spec_v1 = bpf_bypass_spec_v1(NULL);
	u32 elem_size, index_mask, max_entries;
	struct bpf_heap *heap;
	u64 heap_size, mask64;
	u64 heap_value;
	void *value;

	elem_size = round_up(attr->value_size, 8);

	max_entries = attr->max_entries;

	/* On 32 bit archs roundup_pow_of_two() with max_entries that has
	 * upper most bit set in u32 space is undefined behavior due to
	 * resulting 1U << 32, so do it manually here in u64 space.
	 */
	mask64 = fls_long(max_entries - 1);
	mask64 = 1ULL << mask64;
	mask64 -= 1;

	index_mask = mask64;
	if (!bypass_spec_v1) {
		/* round up heap size to nearest power of 2,
		 * since cpu will speculate within index_mask limits
		 */
		max_entries = index_mask + 1;
		/* Check for overflows. */
		if (max_entries < attr->max_entries)
			return ERR_PTR(-E2BIG);
	}

	/* Page align stuff regardless of whether we mmap or not. */
	heap_size = PAGE_ALIGN((u64) max_entries * elem_size);

	if (!is_power_of_2(heap_size))
		return ERR_PTR(-ENOENT);

	heap = bpf_map_area_alloc(sizeof(*heap), numa_node);
	if (!heap)
		return ERR_PTR(-ENOMEM);
	/* allocate all map elements and zero-initialize them */
	if (attr->map_flags & BPF_F_MMAPABLE) {
		/* kmalloc'ed memory can't be mmap'ed, use explicit vmalloc */
		value = bpf_map_area_mmapable_alloc_aligned(heap_size, heap_size, numa_node);
	} else {
		value = bpf_map_area_alloc_aligned(heap_size, heap_size, numa_node);
	}

	if (!value) {
		bpf_map_area_free(heap);
		return ERR_PTR(-ENOMEM);
	}

	// Check alignment constraints
	heap_value = (u64)value;
	/* All bits zero for mask in base addr */
	if (heap_value & (heap_size - 1)) {
		bpf_map_area_free(value);
		bpf_map_area_free(heap);
		return ERR_PTR(-ENOENT);
	}

	heap->value = value;
	heap->index_mask = index_mask;
	heap->map.bypass_spec_v1 = bypass_spec_v1;

	/* copy mandatory map attributes */
	bpf_map_init_from_attr(&heap->map, attr);
	heap->elem_size = elem_size;

	/* Set up the kernel base address for __uptr translation. */
	heap->map.kernel_base_addr = (u64)heap->value;
	heap->map.kernel_addr_mask = (u64)(heap_size - 1);

	return &heap->map;
}

static void *heap_map_elem_ptr(struct bpf_heap* heap, u32 index)
{
	return heap->value + (u64)heap->elem_size * index;
}

/* Called from syscall or from eBPF program */
static void *heap_map_lookup_elem(struct bpf_map *map, void *key)
{
	struct bpf_heap *heap = container_of(map, struct bpf_heap, map);
	u32 index = *(u32 *)key;

	if (unlikely(index >= heap->map.max_entries))
		return NULL;

	return heap->value + (u64)heap->elem_size * (index & heap->index_mask);
}

/* emit BPF instructions equivalent to C code of heap_map_lookup_elem() */
static int heap_map_gen_lookup(struct bpf_map *map, struct bpf_insn *insn_buf)
{
	struct bpf_heap *heap = container_of(map, struct bpf_heap, map);
	struct bpf_insn *insn = insn_buf;
	u32 elem_size = heap->elem_size;
	const int ret = BPF_REG_0;
	const int map_ptr = BPF_REG_1;
	const int index = BPF_REG_2;

	/* Load heap->value into map_ptr reg */
	*insn++ = BPF_LDX_MEM(BPF_DW, map_ptr, map_ptr, offsetof(struct bpf_heap, value));
	*insn++ = BPF_LDX_MEM(BPF_W, ret, index, 0);
	if (!map->bypass_spec_v1) {
		*insn++ = BPF_JMP_IMM(BPF_JGE, ret, map->max_entries, 4);
		*insn++ = BPF_ALU32_IMM(BPF_AND, ret, heap->index_mask);
	} else {
		*insn++ = BPF_JMP_IMM(BPF_JGE, ret, map->max_entries, 3);
	}

	if (is_power_of_2(elem_size)) {
		*insn++ = BPF_ALU64_IMM(BPF_LSH, ret, ilog2(elem_size));
	} else {
		*insn++ = BPF_ALU64_IMM(BPF_MUL, ret, elem_size);
	}
	*insn++ = BPF_ALU64_REG(BPF_ADD, ret, map_ptr);
	*insn++ = BPF_JMP_IMM(BPF_JA, 0, 0, 1);
	*insn++ = BPF_MOV64_IMM(ret, 0);
	return insn - insn_buf;
}

/* Called from syscall */
static int heap_map_get_next_key(struct bpf_map *map, void *key, void *next_key)
{
	struct bpf_heap *heap = container_of(map, struct bpf_heap, map);
	u32 index = key ? *(u32 *)key : U32_MAX;
	u32 *next = (u32 *)next_key;

	if (index >= heap->map.max_entries) {
		*next = 0;
		return 0;
	}

	if (index == heap->map.max_entries - 1)
		return -ENOENT;

	*next = index + 1;
	return 0;
}

/* Called from syscall or from eBPF program */
static long heap_map_update_elem(struct bpf_map *map, void *key, void *value,
				  u64 map_flags)
{
	struct bpf_heap *heap = container_of(map, struct bpf_heap, map);
	u32 index = *(u32 *)key;
	char *val;

	if (unlikely((map_flags & ~BPF_F_LOCK) > BPF_EXIST))
		/* unknown flags */
		return -EINVAL;

	if (unlikely(index >= heap->map.max_entries))
		/* all elements were pre-allocated, cannot insert a new one */
		return -E2BIG;

	if (unlikely(map_flags & BPF_NOEXIST))
		/* all elements already exist */
		return -EEXIST;

	if (unlikely((map_flags & BPF_F_LOCK) &&
		     !btf_record_has_field(map->record, BPF_SPIN_LOCK)))
		return -EINVAL;

	val = heap->value +
		(u64)heap->elem_size * (index & heap->index_mask);
	if (map_flags & BPF_F_LOCK)
		copy_map_value_locked(map, val, value, false);
	else
		copy_map_value(map, val, value);
	bpf_obj_free_fields(heap->map.record, val);
	return 0;
}

/* Called from syscall or from eBPF program */
static long heap_map_delete_elem(struct bpf_map *map, void *key)
{
	return -EINVAL;
}

/* Called when map->refcnt goes to zero, either from workqueue or from syscall */
static void heap_map_free(struct bpf_map *map)
{
	struct bpf_heap *heap = container_of(map, struct bpf_heap, map);

	BUG_ON(!IS_ERR_OR_NULL(map->record));

	bpf_map_area_free(heap->value);
	bpf_map_area_free(heap);
}

static void heap_map_seq_show_elem(struct bpf_map *map, void *key,
				    struct seq_file *m)
{
	void *value;

	rcu_read_lock();

	value = heap_map_lookup_elem(map, key);
	if (!value) {
		rcu_read_unlock();
		return;
	}

	if (map->btf_key_type_id)
		seq_printf(m, "%u: ", *(u32 *)key);
	btf_type_seq_show(map->btf, map->btf_value_type_id, value, m);
	seq_puts(m, "\n");

	rcu_read_unlock();
}

static int heap_map_check_btf(const struct bpf_map *map,
			       const struct btf *btf,
			       const struct btf_type *key_type,
			       const struct btf_type *value_type)
{
	u32 int_data;

	/* One exception for keyless BTF: .bss/.data/.rodata map */
	if (btf_type_is_void(key_type)) {
		if (map->map_type != BPF_MAP_TYPE_HEAP ||
		    map->max_entries != 1)
			return -EINVAL;

		if (BTF_INFO_KIND(value_type->info) != BTF_KIND_DATASEC)
			return -EINVAL;

		return 0;
	}

	if (BTF_INFO_KIND(key_type->info) != BTF_KIND_INT)
		return -EINVAL;

	int_data = *(u32 *)(key_type + 1);
	/* bpf heap can only take a u32 key. This check makes sure
	 * that the btf matches the attr used during map_create.
	 */
	if (BTF_INT_BITS(int_data) != 32 || BTF_INT_OFFSET(int_data))
		return -EINVAL;

	return 0;
}

static int heap_map_mmap(struct bpf_map *map, struct vm_area_struct *vma)
{
	struct bpf_heap *heap = container_of(map, struct bpf_heap, map);
	bool own = true;
	int ret;

	if (!(map->map_flags & BPF_F_MMAPABLE))
		return -EINVAL;

	if (vma->vm_pgoff * PAGE_SIZE + (vma->vm_end - vma->vm_start) >
	    PAGE_ALIGN((u64)heap->map.max_entries * heap->elem_size))
		return -EINVAL;

	/* Ensure only one mmap operation wins in setting base address. */
	if (!cmpxchg64(&heap->map.user_base_addr, 0, vma->vm_start))
		own = false;

	ret = remap_vmalloc_range(vma, heap->value, vma->vm_pgoff);
	/* Undo cmpxchg, as remap_valloc_range failed. */
	if (ret && own)
		heap->map.user_base_addr = 0;
	return ret;
}

struct bpf_iter_seq_heap_map_info {
	struct bpf_map *map;
	u32 index;
};

static void *bpf_heap_map_seq_start(struct seq_file *seq, loff_t *pos)
{
	struct bpf_iter_seq_heap_map_info *info = seq->private;
	struct bpf_map *map = info->map;
	struct bpf_heap *heap;
	u32 index;

	if (info->index >= map->max_entries)
		return NULL;

	if (*pos == 0)
		++*pos;
	heap = container_of(map, struct bpf_heap, map);
	index = info->index & heap->index_mask;
	return heap_map_elem_ptr(heap, index);
}

static void *bpf_heap_map_seq_next(struct seq_file *seq, void *v, loff_t *pos)
{
	struct bpf_iter_seq_heap_map_info *info = seq->private;
	struct bpf_map *map = info->map;
	struct bpf_heap *heap;
	u32 index;

	++*pos;
	++info->index;
	if (info->index >= map->max_entries)
		return NULL;

	heap = container_of(map, struct bpf_heap, map);
	index = info->index & heap->index_mask;
	return heap_map_elem_ptr(heap, index);
}

static int __bpf_heap_map_seq_show(struct seq_file *seq, void *v)
{
	struct bpf_iter_seq_heap_map_info *info = seq->private;
	struct bpf_iter__bpf_map_elem ctx = {};
	struct bpf_iter_meta meta;
	struct bpf_prog *prog;

	meta.seq = seq;
	prog = bpf_iter_get_info(&meta, v == NULL);
	if (!prog)
		return 0;

	ctx.meta = &meta;
	ctx.map = info->map;
	if (v) {
		ctx.key = &info->index;
		ctx.value = v;
	}

	return bpf_iter_run_prog(prog, &ctx);
}

static int bpf_heap_map_seq_show(struct seq_file *seq, void *v)
{
	return __bpf_heap_map_seq_show(seq, v);
}

static void bpf_heap_map_seq_stop(struct seq_file *seq, void *v)
{
	if (!v)
		(void)__bpf_heap_map_seq_show(seq, NULL);
}

static int bpf_iter_init_heap_map(void *priv_data,
				   struct bpf_iter_aux_info *aux)
{
	struct bpf_iter_seq_heap_map_info *seq_info = priv_data;
	struct bpf_map *map = aux->map;

	/* bpf_iter_attach_map() acquires a map uref, and the uref may be
	 * released before or in the middle of iterating map elements, so
	 * acquire an extra map uref for iterator.
	 */
	bpf_map_inc_with_uref(map);
	seq_info->map = map;
	return 0;
}

static void bpf_iter_fini_heap_map(void *priv_data)
{
	struct bpf_iter_seq_heap_map_info *seq_info = priv_data;

	bpf_map_put_with_uref(seq_info->map);
}

static const struct seq_operations bpf_heap_map_seq_ops = {
	.start	= bpf_heap_map_seq_start,
	.next	= bpf_heap_map_seq_next,
	.stop	= bpf_heap_map_seq_stop,
	.show	= bpf_heap_map_seq_show,
};

static const struct bpf_iter_seq_info iter_seq_info = {
	.seq_ops		= &bpf_heap_map_seq_ops,
	.init_seq_private	= bpf_iter_init_heap_map,
	.fini_seq_private	= bpf_iter_fini_heap_map,
	.seq_priv_size		= sizeof(struct bpf_iter_seq_heap_map_info),
};

static long bpf_for_each_heap_elem(struct bpf_map *map, bpf_callback_t callback_fn,
				    void *callback_ctx, u64 flags)
{
	u32 i, key, num_elems = 0;
	struct bpf_heap *heap;
	u64 ret = 0;
	void *val;

	if (flags != 0)
		return -EINVAL;

	heap = container_of(map, struct bpf_heap, map);
	for (i = 0; i < map->max_entries; i++) {
		val = heap_map_elem_ptr(heap, i);
		num_elems++;
		key = i;
		ret = callback_fn((u64)(long)map, (u64)(long)&key,
				  (u64)(long)val, (u64)(long)callback_ctx, 0);
		/* return value: 0 - continue, 1 - stop and return */
		if (ret)
			break;
	}
	return num_elems;
}

static u64 heap_map_mem_usage(const struct bpf_map *map)
{
	struct bpf_heap *heap = container_of(map, struct bpf_heap, map);
	u32 elem_size = heap->elem_size;
	u64 entries = map->max_entries;

	return sizeof(*heap) + PAGE_ALIGN(entries * elem_size);
}

BTF_ID_LIST_SINGLE(heap_map_btf_ids, struct, bpf_heap)
const struct bpf_map_ops heap_map_ops = {
	.map_meta_equal = bpf_map_meta_equal,
	.map_alloc_check = heap_map_alloc_check,
	.map_alloc = heap_map_alloc,
	.map_free = heap_map_free,
	.map_get_next_key = heap_map_get_next_key,
	.map_lookup_elem = heap_map_lookup_elem,
	.map_update_elem = heap_map_update_elem,
	.map_delete_elem = heap_map_delete_elem,
	.map_gen_lookup = heap_map_gen_lookup,
	.map_mmap = heap_map_mmap,
	.map_seq_show_elem = heap_map_seq_show_elem,
	.map_check_btf = heap_map_check_btf,
	.map_lookup_batch = generic_map_lookup_batch,
	.map_update_batch = generic_map_update_batch,
	.map_set_for_each_callback_args = map_set_for_each_callback_args,
	.map_for_each_callback = bpf_for_each_heap_elem,
	.map_mem_usage = heap_map_mem_usage,
	.map_btf_id = &heap_map_btf_ids[0],
	.iter_seq_info = &iter_seq_info,
};
